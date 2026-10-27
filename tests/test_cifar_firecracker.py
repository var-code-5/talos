"""Opt-in real CIFAR guest comparison, including the large tensor transport."""
import importlib.util
import os
from pathlib import Path
import unittest

READY=(os.environ.get('MTCP_TEST_FIRECRACKER')=='1' and importlib.util.find_spec('torch') is not None
       and Path('.mtcp/firecracker-cifar/manifest.json').exists())


@unittest.skipUnless(READY,'Prepare CIFAR Firecracker assets and set MTCP_TEST_FIRECRACKER=1')
class CifarGuestTests(unittest.TestCase):
    def test_real_pytorch_guest_matches_host_round(self):
        import numpy as np
        from mtcp.cifar.training import train,model,weights,decode
        from mtcp.cifar.data import load_manifest
        from mtcp.firecracker import Firecracker
        data='.mtcp/cifar10'
        payload=dict(dataset_id=load_manifest(data)['dataset_id'],seed=42,threads=1,
            train_limit=128,peers=2,peer=0,epoch=0,start=0,count=64,batch_size=32,
            lr=.001,momentum=.9,weights=weights(model()),optimizer_momentum=None)
        host=train(payload,data)
        runner=Firecracker('.mtcp/firecracker-cifar',memory_mb=768)
        guest=runner.execute(payload,operation='cifar-train',timeout=90,memory_mb=768)
        self.assertTrue(guest['fresh_vm']);self.assertEqual(guest['isolation'],'kvm-microvm')
        self.assertEqual(guest['count'],64)
        self.assertTrue(np.allclose(decode(host['weights']),decode(guest['weights']),atol=1e-6,rtol=1e-5))
        self.assertTrue(np.allclose(decode(host['optimizer_momentum']),decode(guest['optimizer_momentum']),atol=1e-6,rtol=1e-5))
        # The next task transfers both weights and momentum (>230 KiB) through vsock.
        payload.update(weights=guest['weights'],optimizer_momentum=guest['optimizer_momentum'],epoch=1)
        next_host=train(payload,data)
        next_guest=runner.execute(payload,operation='cifar-train',timeout=90,memory_mb=768)
        self.assertTrue(np.allclose(decode(next_host['weights']),decode(next_guest['weights']),atol=1e-6,rtol=1e-5))
        payload['numerics']='fp64-portable'
        portable_host=train(payload,data)
        portable_guest=runner.execute(payload,operation='cifar-train',timeout=90,memory_mb=768)
        self.assertEqual(portable_guest['compute_dtype'],'torch.float64')
        self.assertFalse(portable_guest['mkldnn_enabled'])
        self.assertTrue(np.allclose(decode(portable_host['weights']),decode(portable_guest['weights']),atol=1e-6,rtol=1e-5))
        self.assertTrue(np.allclose(decode(portable_host['optimizer_momentum']),decode(portable_guest['optimizer_momentum']),atol=1e-6,rtol=1e-5))

    def test_version2_guest_range_and_optimizer_replay(self):
        import hashlib
        import numpy as np
        from mtcp.kernel import digest
        from mtcp.cifar.allocation import SETTINGS,make_plan
        from mtcp.cifar.training import train,model,weights,decode,ARTIFACT
        from mtcp.cifar.data import load_manifest
        from mtcp.firecracker import Firecracker
        initial=weights(model(numerics='fp64-portable'))
        config=dict(run_id='guestallocation',members=[{'id':'a'},{'id':'b'}],train_limit=96,
            sync_samples=96,batch_size=16,allocation='compute-aware',audit='none',planner=SETTINGS)
        c=lambda us:dict(train_us=us,audit_us=us,setup_us=0,audit_setup_us=0,evaluation_us=0)
        plan=make_plan(config,0,0,0,hashlib.sha256(decode(initial).tobytes()).hexdigest(),[c(1),c(2)],[])
        payload=dict(dataset_id=load_manifest('.mtcp/cifar10')['dataset_id'],seed=42,threads=1,
            train_limit=96,peers=2,peer=0,epoch=0,start=0,count=plan['counts'][0],batch_size=16,
            lr=.001,momentum=.9,weights=initial,optimizer_momentum=None,numerics='fp64-portable',
            indexing_version=2,plan=plan,plan_digest=digest(plan),run_id=config['run_id'],config_digest=digest(config),round=0)
        runner=Firecracker('.mtcp/firecracker-cifar',memory_mb=768)
        for epoch in (0,1):
            plan['epoch']=epoch;plan['round']=epoch;plan['model_digest']=hashlib.sha256(decode(payload['weights']).tobytes()).hexdigest()
            payload.update(epoch=epoch,round=epoch,plan_digest=digest(plan))
            host=train(payload,'.mtcp/cifar10')
            guest=runner.execute(payload,operation='cifar-train',timeout=120,memory_mb=768)
            for key in ('weights','optimizer_momentum'):
                self.assertTrue(np.allclose(decode(host[key]),decode(guest[key]),atol=1e-6,rtol=1e-5))
            payload.update(weights=guest['weights'],optimizer_momentum=guest['optimizer_momentum'])
