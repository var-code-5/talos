import importlib.util
import os
from pathlib import Path
import unittest

HAS_TORCH=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None


@unittest.skipUnless(HAS_TORCH,'Install requirements-cifar.txt for CIFAR tests')
class CIFARTests(unittest.TestCase):
    def test_exact_project1_model_and_safe_tensor_encoding(self):
        import numpy as np
        from mtcp.cifar.training import model,weights,decode,encode,PARAMETERS,set_weights
        net=model();encoded=weights(net)
        self.assertEqual(sum(p.numel() for p in net.parameters()),PARAMETERS)
        self.assertEqual(len(decode(encoded)),22058)
        self.assertEqual(encode(decode(encoded)),encoded)
        other=model(seed=1);set_weights(other,encoded)
        self.assertEqual(weights(other),encoded)
        with self.assertRaises(ValueError): decode('AAAA')
        bad=np.zeros(PARAMETERS);bad[0]=float('nan')
        with self.assertRaises(ValueError):encode(bad)

    def test_disjoint_reproducible_epoch_shards(self):
        import numpy as np
        from mtcp.cifar.training import indices
        a=indices(42,1000,2,0,0,0,500);b=indices(42,1000,2,1,0,0,500)
        self.assertEqual(len(set(a)&set(b)),0)
        self.assertEqual(len(set(a)|set(b)),1000)
        self.assertTrue(np.array_equal(a,indices(42,1000,2,0,0,0,500)))
        other=indices(42,1000,2,0,1,0,500)
        self.assertEqual(set(a),set(other));self.assertFalse(np.array_equal(a,other))

    @unittest.skipUnless(Path(os.environ.get('MTCP_CIFAR_DATA','.mtcp/cifar10'),'manifest.json').exists(),'Prepare real CIFAR data first')
    def test_real_data_deterministic_training_and_evaluation(self):
        from mtcp.cifar.data import load_manifest
        from mtcp.cifar.training import model,weights,train,evaluate
        directory=os.environ.get('MTCP_CIFAR_DATA','.mtcp/cifar10')
        dataset=load_manifest(directory,verify=True)
        payload=dict(dataset_id=dataset['dataset_id'],seed=42,threads=1,train_limit=128,
            peers=2,peer=0,epoch=0,start=0,count=64,batch_size=32,lr=.001,momentum=.9,
            weights=weights(model()),optimizer_momentum=None)
        a=train(payload,directory);b=train(payload,directory)
        self.assertEqual(a['weights'],b['weights']);self.assertEqual(a['optimizer_momentum'],b['optimizer_momentum'])
        self.assertEqual(a['count'],64)
        self.assertNotEqual(a['weights'],payload['weights'])
        evaluation=evaluate(a['weights'],directory,test_limit=64)
        self.assertEqual(sum(map(sum,evaluation['confusion_matrix'])),64)
        self.assertTrue(0<=evaluation['test_accuracy']<=1)
        self.assertTrue(0<=evaluation['macro_f1']<=1)

    def test_audit_rejects_corruption_and_reports_error_magnitude(self):
        import numpy as np
        from mtcp.cifar.audit import compare_update
        from mtcp.cifar.training import encode,PARAMETERS
        values=np.linspace(-.1,.1,PARAMETERS,dtype=np.float32)
        reference={'weights':encode(values),'optimizer_momentum':encode(values)}
        self.assertTrue(compare_update(reference,reference)['passed'])
        corrupted=values.copy();corrupted[123]+=.01
        actual=dict(reference,weights=encode(corrupted))
        evidence=compare_update(reference,actual)
        self.assertFalse(evidence['passed'])
        self.assertEqual(evidence['weights']['mismatched_coordinates'],1)
        self.assertGreater(evidence['weights']['max_abs'],.009)
        self.assertTrue(evidence['optimizer_momentum']['passed'])

    @unittest.skipUnless(Path('.mtcp/cifar10/manifest.json').exists(),'Requires prepared real CIFAR data')
    def test_portable_profile_replay_across_cpu_thread_counts(self):
        from mtcp.cifar.data import load_manifest
        from mtcp.cifar.training import model,weights,train
        from mtcp.cifar.audit import compare_update
        payload=dict(dataset_id=load_manifest('.mtcp/cifar10')['dataset_id'],seed=42,threads=1,
            train_limit=10000,peers=2,peer=0,epoch=0,start=0,count=5000,batch_size=32,
            lr=.001,momentum=.9,weights=weights(model()),optimizer_momentum=None,numerics='fp64-portable')
        one=train(payload,'.mtcp/cifar10')
        two=train(dict(payload,threads=2),'.mtcp/cifar10')
        evidence=compare_update(one,two)
        self.assertTrue(evidence['passed'],evidence)
        self.assertEqual(one['compute_dtype'],'torch.float64')
        self.assertFalse(one['mkldnn_enabled'])
        # Restoring the float32 wire momentum must promote it back into FP64 computation.
        payload.update(weights=one['weights'],optimizer_momentum=one['optimizer_momentum'],epoch=1)
        a=train(payload,'.mtcp/cifar10');b=train(dict(payload,threads=2),'.mtcp/cifar10')
        evidence=compare_update(a,b)
        self.assertTrue(evidence['passed'],evidence)
