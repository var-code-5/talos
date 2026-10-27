import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HAS_DATA=Path(os.environ.get('MTCP_CIFAR_DATA','.mtcp/cifar10'),'manifest.json').exists()
HAS_TORCH=importlib.util.find_spec('torch') is not None


def port(kind=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET,kind) as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


@unittest.skipUnless(HAS_DATA and HAS_TORCH,'Requires prepared CIFAR and requirements-cifar.txt')
class PeerTests(unittest.TestCase):
    def test_idle_advertisement_tracks_address_changes(self):
        from unittest.mock import patch
        from mtcp.cifar.peer import Peer
        peer=object.__new__(Peer)
        peer.auto_advertise=True;peer.http_port=9101;peer.active=None
        peer.endpoint='http://172.16.207.7:9101'
        with patch('mtcp.cifar.peer.discover_host',return_value='172.16.203.242'):
            peer.refresh_endpoint()
            self.assertEqual(peer.endpoint,'http://172.16.203.242:9101')
            peer.active='running-experiment';peer.endpoint='http://172.16.207.7:9101'
            peer.refresh_endpoint()
            self.assertEqual(peer.endpoint,'http://172.16.207.7:9101')
            peer.active=None;peer.auto_advertise=False
            peer.refresh_endpoint()
            self.assertEqual(peer.endpoint,'http://172.16.207.7:9101')

    def test_symmetric_discovery_training_and_report(self):
        from argparse import Namespace
        from mtcp.cifar.experiment import start,public,collect
        from mtcp.cifar.network import call
        from mtcp.cifar.report import compare
        with tempfile.TemporaryDirectory() as temporary:
            key=secrets.token_hex(32);env=dict(os.environ,MTCP_CLUSTER_KEY=key)
            discovery=port(socket.SOCK_DGRAM);processes=[];logs=[];endpoints=[]
            from mtcp.host import available_cpus
            cpus=available_cpus()
            old_key=os.environ.get('MTCP_CLUSTER_KEY');os.environ['MTCP_CLUSTER_KEY']=key
            try:
                for i in range(2):
                    number=port();endpoints.append(f'http://127.0.0.1:{number}')
                    log=open(Path(temporary)/f'peer-{i}.log','w+');logs.append(log)
                    processes.append(subprocess.Popen([sys.executable,'-m','mtcp.cifar','peer',
                        '--port',str(number),'--advertise-host','127.0.0.1','--name',f'test-peer-{i}',
                        '--cpus',str(cpus[i%len(cpus)]),'--discovery-port',str(discovery),
                        '--state-dir',str(Path(temporary)/f'peer-{i}'),
                        '--data',os.environ.get('MTCP_CIFAR_DATA','.mtcp/cifar10')],env=env,stdout=log,stderr=log))
                for _ in range(200):
                    try:
                        if all(len(public(e,'/v1/peers')['peers'])==2 for e in endpoints):break
                    except OSError:pass
                    time.sleep(.1)
                else:
                    for log in logs:log.seek(0);print(log.read())
                    self.fail('Equal peers did not discover each other')
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    call(endpoints[0],'/v1/runs','wrong-key',{})
                self.assertEqual(denied.exception.code,401);denied.exception.close()
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    call(endpoints[0],'/v1/calibrate','wrong-key',{'config':{}})
                self.assertEqual(denied.exception.code,401);denied.exception.close()
                args=Namespace(endpoint=endpoints[1],peers=2,discovery_wait=5,epochs=2,
                    train_limit=128,test_limit=64,batch_size=16,sync_samples=64,threads=1,
                    seed=42,lr=.001,momentum=.9,audit='full',no_wait=False,output=str(Path(temporary)/'results'))
                artifact=start(args);result=json.loads(Path(artifact).read_text())
                self.assertTrue(result['peer_models_identical']);self.assertEqual(result['topology'],'single-host-peers')
                self.assertIn('initial_weights',result['config'])
                self.assertEqual(result['config']['numerics'],'fp64-portable')
                self.assertEqual(len({p['initial_weight_digest'] for p in result['peers']}),1)
                for peer in result['peers']:
                    self.assertEqual(peer['totals']['samples'],128)
                    self.assertEqual(len(peer['rounds']),4)
                    self.assertGreater(peer['rtt_ms']['mean'],0)
                    self.assertGreater(peer['totals']['network_response_body_bytes'],0)
                    self.assertGreater(peer['totals']['audit_seconds'],0)
                    self.assertEqual(peer['final_evaluation']['test_count'],64)
                    self.assertTrue(peer['last_audit']['passed'])
                # Proposal retries are idempotent and do not repeat training.
                again,_=call(endpoints[0],'/v1/runs',key,result['config'])
                self.assertEqual(again['status'],'completed')
                collected=subprocess.run([sys.executable,'-m','mtcp.cifar','collect','--endpoint',endpoints[0],
                    '--run-id',result['config']['run_id'],'--output',str(Path(temporary)/'collected')],
                    env=env,capture_output=True,text=True,timeout=15)
                self.assertEqual(collected.returncode,0,collected.stderr)
                report=compare([artifact],str(Path(temporary)/'comparison'))
                self.assertIn('Physical two-host results are pending',Path(report).read_text())
                bad=json.loads(Path(artifact).read_text());bad['config']['batch_size']=32
                bad_path=Path(temporary)/'bad.json';bad_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):compare([artifact,str(bad_path)],str(Path(temporary)/'bad-comparison'))
                bad=json.loads(Path(artifact).read_text())
                for peer in bad['peers']:peer['initial_weight_digest']='0'*64
                bad_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):compare([artifact,str(bad_path)],str(Path(temporary)/'bad-initial'))
                bad=json.loads(Path(artifact).read_text());bad['config']['members'][0]['torch_version']='0.0.0'
                bad_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):compare([artifact,str(bad_path)],str(Path(temporary)/'bad-framework'))
                bad=json.loads(Path(artifact).read_text());bad['config']['numerics']='fp32-native'
                bad_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):compare([artifact,str(bad_path)],str(Path(temporary)/'bad-numerics'))
                # Version-2 equal and measured allocation exercise the actual
                # calibration endpoint, exact replay and all-peer agreement.
                args.epochs=2;args.train_limit=65;args.sync_samples=64
                args.allocation='equal';args.indexing_version=2
                control=start(args)
                args.allocation='compute-aware'
                args.no_wait=True
                proposed=start(args)
                value,_=call(endpoints[0],f'/v1/runs/{proposed}',key)
                measured=collect(value['config'],key,args.output)
                control_run=json.loads(Path(control).read_text())
                measured_run=json.loads(Path(measured).read_text())
                self.assertTrue(measured_run['peer_models_identical'])
                self.assertGreater(measured_run['end_to_end_seconds'],measured_run['wall_seconds'])
                for node in measured_run['peers']:
                    self.assertEqual(len(node['rounds']),4)
                    self.assertTrue(node['last_audit']['passed'])
                    self.assertEqual(sum(r['global_samples'] for r in node['rounds']),130)
                    self.assertTrue(any(r['local_samples']==0 for r in measured_run['peers'][0]['rounds']) or
                                    any(r['local_samples']==0 for r in measured_run['peers'][1]['rounds']))
                with self.assertRaises(ValueError):compare([control,measured],str(Path(temporary)/'policy-mismatch'))
                comparison=compare([control,measured],str(Path(temporary)/'allocation'),allocation_comparison=True)
                self.assertIn('Unequal local SGD',Path(comparison).read_text())
                mismatch=json.loads(Path(measured).read_text())
                mismatch['config']['members'][0]['resources']['cpu_affinity']=[99999]
                changed=Path(temporary)/'changed-hardware.json';changed.write_text(json.dumps(mismatch))
                with self.assertRaises(ValueError):compare([control,str(changed)],str(Path(temporary)/'hardware-mismatch'),allocation_comparison=True)
                from mtcp.cifar.peer import Peer
                restored=Peer(Path(temporary)/'peer-0',os.environ.get('MTCP_CIFAR_DATA','.mtcp/cifar10'),key,'restored',[cpus[0]])
                self.assertEqual(restored.runs[proposed]['assignment']['plan_digest'],measured_run['peers'][0]['rounds'][-1]['plan_digest'])
                body=public(endpoints[0],'/v1/status')
                self.assertEqual(body['runs'][0]['allocation'],'compute-aware')
                # Stored local calibration prevents replacement by the proposer.
                forged=json.loads(Path(measured).read_text())['config']
                forged['calibration'][0]['cost']['train_us']+=1
                with self.assertRaises(urllib.error.HTTPError):call(forged['members'][0]['endpoint'],'/v1/runs',key,forged)
                status=public(endpoints[0],'/v1/status')
                self.assertEqual(status['runs'][0]['rounds'],4)
                with urllib.request.urlopen(endpoints[0]) as response:
                    self.assertIn(b'Every peer trains.',response.read())
            finally:
                for process in processes:process.terminate()
                for process in processes:process.wait(timeout=10)
                for log in logs:log.close()
                if old_key is None:os.environ.pop('MTCP_CLUSTER_KEY',None)
                else:os.environ['MTCP_CLUSTER_KEY']=old_key
