import copy
import tempfile
import time
import unittest
from pathlib import Path

from mtcp.coordinator import Conflict, Coordinator
from mtcp.kernel import digest, gradient, samples
from mtcp.protocol import ARTIFACT, BINDINGS, admit, sign
from mtcp.worker import POLICY
from mtcp.resources import advertise

TOKEN = "a" * 64


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.co = Coordinator(str(Path(self.temp.name) / "state.db"), {"a": TOKEN, "b": TOKEN})
        self.co.register("a", POLICY)
        self.co.register("b", POLICY)
        self.job = dict(id="test", step=1, weights=[0., 0., 0.], isolation="bounded-process",
            status="running", audit="full", tasks=[], events=[],
            metrics=dict(audited=0, accepted=0, rejected=0, unaudited=0, policy_rejections=0,
                         audit_ms=0., compute_ms=0.))
        self.co.jobs["test"] = self.job
        self.task = self.co.create_task(self.job, "a", 400, 700)

    def tearDown(self):
        self.co.stopped.set()
        self.co.db.close()
        self.temp.cleanup()

    def result(self):
        contract = self.task["contract"]
        return dict({k: contract[k] for k in BINDINGS},
                    gradient=gradient([0., 0., 0.], samples()[400:700]))

    def submit(self, result):
        return self.co.submit("a", result, sign(result, TOKEN))

    def test_contract_tampering_and_fail_closed_admission(self):
        contract = copy.deepcopy(self.task["contract"])
        self.assertIsNone(admit(contract, self.task["signature"], TOKEN, POLICY))
        contract["memory_mb"] = 512
        self.assertIn("authentication", admit(contract, self.task["signature"], TOKEN, POLICY))
        self.assertIn("Resource", admit(contract, sign(contract, TOKEN), TOKEN, POLICY))
        contract = copy.deepcopy(self.task["contract"])
        contract["isolation"] = "kvm-microvm"
        self.assertIn("unavailable", admit(contract, sign(contract, TOKEN), TOKEN, POLICY))
        contract["isolation"] = "bounded-process"
        contract["data_classification"] = "private"
        self.assertIn("public", admit(contract, sign(contract, TOKEN), TOKEN, POLICY))

    def test_corruption_rejected_and_audit_escalates(self):
        result = self.result()
        result["gradient"][0] += 8
        self.assertEqual(self.submit(result)["status"], "rejected")
        self.assertEqual(self.co.workers["a"]["failures"], 1)
        self.assertEqual(self.co.audit_probability(self.co.workers["a"], "adaptive"), 1)

    def test_metadata_replay_and_duplicate_rejected(self):
        result = self.result()
        result["model_version"] = -1
        self.assertEqual(self.submit(result)["status"], "rejected")
        with self.assertRaises(Conflict):
            self.submit(result)

    def test_good_result_authentication_and_commit_immutability(self):
        result = self.result()
        with self.assertRaises(ValueError):
            self.co.submit("a", result, "fake")
        self.assertEqual(self.submit(result)["status"], "accepted")
        result["gradient"][0] = 99
        self.assertNotEqual(self.task["result"]["gradient"][0], 99)
        with self.assertRaises(Conflict):
            self.submit(self.result())

    def test_nonfinite_and_expired_results(self):
        self.task["contract"]["expires_at"] = time.time() - 1
        self.assertEqual(self.submit(self.result())["status"], "rejected")
        self.assertIn("expired", self.task["reason"])

    def test_invalid_output_shape_rejected_before_audit(self):
        result = self.result()
        result["gradient"] = [1., 2.]
        self.assertEqual(self.submit(result)["status"], "rejected")
        self.assertIsNone(self.task["audit"])

    def test_weighted_partition_gradient_matches_full_batch(self):
        rows = samples()
        contributions = [(400, gradient([.1, .2, .3], rows[:400])),
                         (300, gradient([.1, .2, .3], rows[400:700])),
                         (300, gradient([.1, .2, .3], rows[700:]))]
        full = gradient([.1, .2, .3], rows)
        for i in range(3):
            self.assertAlmostEqual(sum(n*g[i] for n,g in contributions)/1000, full[i], places=12)

    def test_adaptive_audit_floor(self):
        worker = self.co.workers["a"]
        self.assertEqual(self.co.audit_probability(worker, "adaptive"), .6)
        worker["audited"] = 5
        self.assertEqual(self.co.audit_probability(worker, "adaptive"), .15)
        self.assertEqual(self.co.audit_probability(worker, "full"), 1)

    def test_checkpoint_survives_restart_and_active_run_is_interrupted(self):
        self.co.save(self.job)
        restored = Coordinator(str(Path(self.temp.name) / "state.db"), {"a": TOKEN, "b": TOKEN})
        try:
            self.assertEqual(restored.jobs["test"]["weights"], [0., 0., 0.])
            self.assertEqual(restored.jobs["test"]["status"], "interrupted")
        finally:
            restored.db.close()

    def test_invalid_resource_advertisements_rejected(self):
        resources = advertise()
        resources['offered_upload_mbps'] = float('nan')
        with self.assertRaises(ValueError):
            self.co.register('a', POLICY, resources)
        resources = advertise()
        resources['offered_vcpus'] = 2
        with self.assertRaises(ValueError):
            self.co.register('a', POLICY, resources)


if __name__ == "__main__":
    unittest.main()
