"""Allocation integrity and calibration failure cases, independent of LAN access."""
import copy
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mtcp.kernel import digest
from mtcp.cifar.allocation import SETTINGS,make_plan,smooth
from test_cifar_allocation import cost


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mtcp.cifar.training import model,weights,ARTIFACT
        cls.config=dict(run_id='allocationtest',members=[dict(id='a')],epochs=1,train_limit=64,
            test_limit=16,batch_size=16,sync_samples=64,threads=1,seed=42,lr=.001,momentum=.9,
            audit='full',numerics='fp64-portable',dataset_id='data',artifact_digest=ARTIFACT,
            initial_weights=weights(model()),allocation='compute-aware',indexing_version=2,planner=SETTINGS)

    def test_plan_rejects_wrong_range_count_model_and_digest(self):
        from mtcp.cifar.training import validate_plan,decode
        from mtcp.cifar.execution import model_digest
        c=self.config;md=model_digest(c['initial_weights'])
        plan=make_plan(c,0,0,0,md,[cost(100)],[])
        payload=dict(peers=1,peer=0,count=64,start=0,train_limit=64,run_id=c['run_id'],
            config_digest=digest(c),epoch=0,round=0,plan_digest=digest(plan))
        validate_plan(plan,payload,md)
        for key,value in [('count',63),('start',1),('round',1),('plan_digest','wrong')]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_plan(plan,dict(payload,**{key:value}),md)
        with self.assertRaises(ValueError):validate_plan(plan,payload,'wrong-model')

    def test_unsupported_peer_is_rejected_before_calibration(self):
        from mtcp.cifar.experiment import start
        member=dict(id='a',endpoint='http://unused')
        args=SimpleNamespace(discovery_wait=1,endpoint='http://unused',peers=1,allocation='compute-aware',indexing_version=None)
        with patch.dict('os.environ',{'MTCP_CLUSTER_KEY':'x'*32}),patch('mtcp.cifar.experiment.public',return_value={'peers':[member]}),patch('mtcp.cifar.experiment.call',return_value=({'peer':member},{})),self.assertRaisesRegex(ValueError,'Update and restart'):
            start(args)

    def test_calibration_isolation_busy_invalid_and_timeout(self):
        from mtcp.cifar.calibration import calibrate,isolated
        peer=SimpleNamespace(lock=threading.RLock(),active=None,threads=1,manifest={'dataset_id':'data'},
            runner=None,data='unused',capacity={'cpu_affinity':[0]},id='a',calibrations={})
        result=dict(training=[dict(count=64,train_seconds=.064,wall_seconds=.065) for _ in range(4)],
                    evaluation=dict(evaluation_seconds=.01))
        before=copy.deepcopy(self.config)
        with patch('mtcp.cifar.calibration.isolated',return_value=result):
            record=calibrate(peer,dict(config=self.config))
        self.assertEqual(self.config,before);self.assertIsNone(peer.active)
        self.assertEqual(record['cost']['train_us'],1000)
        self.assertEqual(peer.calibrations[digest(self.config)],record)
        peer.active='other'
        with self.assertRaises(ValueError):calibrate(peer,dict(config=self.config))
        peer.active=None
        result['training'][1]['train_seconds']=float('nan')
        with patch('mtcp.cifar.calibration.isolated',return_value=result),self.assertRaises(ValueError):calibrate(peer,dict(config=self.config))
        self.assertIsNone(peer.active)
        with patch('subprocess.run',side_effect=subprocess.TimeoutExpired('worker',120)),self.assertRaisesRegex(ValueError,'deadline'):
            isolated('train',{},'unused',[0])

    def test_admission_binding_and_round_limits(self):
        from mtcp.cifar.peer import validate_config
        c=copy.deepcopy(self.config)
        validate_config(c,require_calibration=False)
        request=digest(c)
        c.update(calibration=[dict(peer_id='a',request_digest=request,cost=cost(100),seconds=.1)],
                 calibration_request_digest=request,startup_seconds=.2)
        validate_config(c)
        c['lr']=.01
        with self.assertRaisesRegex(ValueError,'changed'):validate_config(c)
        c=copy.deepcopy(self.config);c.update(epochs=100,sync_samples=2)
        with self.assertRaisesRegex(ValueError,'500 rounds'):validate_config(c,require_calibration=False)
        c=copy.deepcopy(self.config);c['indexing_version']=1
        with self.assertRaises(ValueError):validate_config(c,require_calibration=False)

    def test_immutable_evidence_and_zero_rate_retention(self):
        from mtcp.cifar.execution import publish,observation
        peer=SimpleNamespace(lock=threading.RLock());run={'plans':{}}
        publish(peer,run,'plans',0,{'a':1});publish(peer,run,'plans',0,{'a':1})
        with self.assertRaises(ValueError):publish(peer,run,'plans',0,{'a':2})
        previous=cost(100)
        actual=observation(dict(count=0),0,0,0,0,0,dict(evaluation_seconds=.01),previous)
        self.assertEqual(actual['train_us'],100);self.assertEqual(actual['audit_us'],100)
        # Same authenticated observations always produce identical smoothing.
        seq=[cost(n) for n in (200,400,10000,50,100)]
        a=b=previous
        for obs in seq:a=smooth(a,obs);b=smooth(b,obs)
        self.assertEqual(a,b)

    @unittest.skipUnless(Path('.mtcp/cifar10/manifest.json').exists(),'Requires prepared CIFAR')
    def test_real_unequal_ranges_replay_with_momentum(self):
        import numpy as np
        from mtcp.cifar.training import train,decode,encode,PARAMETERS
        from mtcp.cifar.execution import model_digest
        from mtcp.cifar.data import load_manifest
        c=copy.deepcopy(self.config);c['members']=[{'id':'a'},{'id':'b'}]
        initial=c['initial_weights'];momenta=[None,None];replay_momenta=[None,None]
        for epoch,counts in enumerate(([48,16],[16,48])):
            plan=make_plan(c,epoch,epoch,0,model_digest(initial),[cost(1),cost(2)],[])
            plan['counts']=list(counts)
            updates=[]
            for i,count in enumerate(counts):
                payload=dict(dataset_id=load_manifest('.mtcp/cifar10')['dataset_id'],seed=42,threads=1,
                    train_limit=64,peers=2,peer=i,epoch=epoch,start=sum(counts[:i]),count=count,
                    batch_size=16,lr=.001,momentum=.9,weights=initial,optimizer_momentum=momenta[i],
                    numerics='fp64-portable',indexing_version=2,plan=plan,plan_digest=digest(plan),
                    run_id=c['run_id'],config_digest=digest(c),round=epoch)
                result=train(payload,'.mtcp/cifar10')
                replay=train(dict(payload,optimizer_momentum=replay_momenta[i]),'.mtcp/cifar10')
                for k in ('weights','optimizer_momentum'):self.assertEqual(result[k],replay[k])
                updates.append(result);momenta[i]=result['optimizer_momentum'];replay_momenta[i]=replay['optimizer_momentum']
            initial=encode(sum(decode(r['weights']).astype(np.float64)*r['count']/64 for r in updates))

    def test_actual_execution_rejects_disagreement_before_training(self):
        import tempfile
        from tools.cifar.validate_allocation import trial
        with tempfile.TemporaryDirectory() as directory:
            nodes=trial('compute-aware',directory,disagree=True)
        self.assertTrue(all(n['status']=='failed' for n in nodes))
        self.assertTrue(all(not n['rounds'] for n in nodes))
        self.assertTrue(all('disagree' in n['error'] or 'failed' in n['error'] for n in nodes))

    def test_tiny_epoch_tail_does_not_reset_next_epoch_shares(self):
        import tempfile
        from tools.cifar.validate_allocation import trial
        with tempfile.TemporaryDirectory() as directory:
            nodes=trial('compute-aware',directory,train_limit=9601,epochs=2)
        self.assertTrue(all(n['status']=='completed' for n in nodes))
        rounds=nodes[0]['rounds']
        self.assertEqual(rounds[0]['assigned_samples'],[6400,3200])
        self.assertEqual(rounds[1]['global_samples'],1)
        self.assertEqual(rounds[2]['assigned_samples'],[6400,3200])

    def test_polling_success_transfer_and_failure(self):
        from mtcp.cifar.execution import Exchange
        peer=SimpleNamespace(stop=threading.Event(),key='key')
        metric=dict(seconds=.001,request_body_bytes=1,response_body_bytes=2)
        values=[({'ready':False},metric),({'ready':True},metric)]
        with patch('mtcp.cifar.execution.call',side_effect=values):
            x=Exchange(peer);x.wait(dict(name='b',endpoint='unused'),'/route',lambda v:v['ready'])
        self.assertGreaterEqual(x.polling_seconds,.04);self.assertEqual(x.transfer_seconds,.001)
        self.assertEqual(x.request_body_bytes,2)
        with patch('mtcp.cifar.execution.call',return_value=({'status':'failed','error':'gone'},metric)),self.assertRaisesRegex(RuntimeError,'gone'):
            Exchange(peer).wait(dict(name='b',endpoint='unused'),'/route',lambda v:False)
        peer.stop.set()
        with self.assertRaises(TimeoutError):Exchange(peer).wait(dict(name='b',endpoint='unused'),'/route',lambda v:False)


if __name__=='__main__':unittest.main()
