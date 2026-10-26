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
