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
    def test_symmetric_discovery_training_and_report(self):
        from argparse import Namespace
        from mtcp.cifar.experiment import start,public
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
                args=Namespace(endpoint=endpoints[1],peers=2,discovery_wait=5,epochs=2,
                    train_limit=128,test_limit=64,batch_size=16,sync_samples=64,threads=1,
                    seed=42,lr=.001,momentum=.9,audit='full',no_wait=False,output=str(Path(temporary)/'results'))
                artifact=start(args);result=json.loads(Path(artifact).read_text())
                self.assertTrue(result['peer_models_identical']);self.assertEqual(result['topology'],'single-host-peers')
                for peer in result['peers']:
                    self.assertEqual(peer['totals']['samples'],128)
                    self.assertEqual(len(peer['rounds']),4)
                    self.assertGreater(peer['rtt_ms']['mean'],0)
                    self.assertGreater(peer['totals']['network_response_body_bytes'],0)
                    self.assertGreater(peer['totals']['audit_seconds'],0)
                    self.assertEqual(peer['final_evaluation']['test_count'],64)
                # Proposal retries are idempotent and do not repeat training.
                again,_=call(endpoints[0],'/v1/runs',key,result['config'])
                self.assertEqual(again['status'],'completed')
                report=compare([artifact],str(Path(temporary)/'comparison'))
                self.assertIn('Physical two-host results are pending',Path(report).read_text())
                bad=json.loads(Path(artifact).read_text());bad['config']['batch_size']=32
                bad_path=Path(temporary)/'bad.json';bad_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):compare([artifact,str(bad_path)],str(Path(temporary)/'bad-comparison'))
            finally:
                for process in processes:process.terminate()
                for process in processes:process.wait(timeout=10)
                for log in logs:log.close()
                if old_key is None:os.environ.pop('MTCP_CLUSTER_KEY',None)
                else:os.environ['MTCP_CLUSTER_KEY']=old_key
