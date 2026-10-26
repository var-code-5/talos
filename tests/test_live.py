"""Integration tests exercise real HTTP, worker controllers, and kernel children."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from mtcp.kernel import gradient, samples
from mtcp.transport import request


class LiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.database = str(Path(cls.temp.name) / "state.db")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{cls.port}"
        cls.process = subprocess.Popen([sys.executable, "-m", "mtcp", "--demo", "--port", str(cls.port),
            "--database", cls.database], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                if len(cls.api("/api/state")["workers"]) == 2:
                    return
            except OSError:
                pass
            time.sleep(.1)
        cls.process.terminate()
        raise RuntimeError("Demo did not start")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=8)
        cls.temp.cleanup()

    @classmethod
    def api(cls, route, data=None, headers=None):
        req = urllib.request.Request(cls.url + route,
            data=json.dumps(data).encode() if data is not None else None,
            headers=headers or {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response)

    def run_job(self, **options):
        self.api("/api/jobs", dict(steps=4, **options))
        for _ in range(200):
            job = self.api("/api/state")["job"]
            if job["status"] != "running":
                self.assertEqual(job["status"], "completed", job["events"])
                return job
            time.sleep(.08)
        self.fail("Run did not complete")

    def assert_matches_reference(self, job):
        weights = [0., 0., 0.]
        for _ in range(4):
            g = gradient(weights, samples())
            weights = [w - .3*v for w,v in zip(weights,g)]
        for expected, actual in zip(weights, job["weights"]):
            self.assertAlmostEqual(expected, actual, places=11)
        self.assertLess(job["history"][-1]["loss"], job["history"][0]["loss"])

    def test_honest_training_and_persistent_export(self):
        job = self.run_job()
        self.assert_matches_reference(job)
        self.assertEqual(job["metrics"]["accepted"], 8)
        self.assertEqual(job["metrics"]["audited"], 8)
        exported = self.api("/api/export")
        self.assertTrue(any(j["id"] == job["id"] for j in exported["jobs"]))

    def test_corruption_and_replay_recover_exactly(self):
        for scenario in ("corrupt", "replay"):
            with self.subTest(scenario=scenario):
                job = self.run_job(scenario=scenario)
                self.assert_matches_reference(job)
                self.assertEqual(job["metrics"]["rejected"], 1)
                self.assertEqual(job["metrics"]["recovered"], 1)

    def test_timeout_recovers(self):
        job = self.run_job(scenario="timeout")
        self.assert_matches_reference(job)
        self.assertEqual(job["metrics"]["timeouts"], 1)
        self.assertEqual(job["metrics"]["recovered"], 1)

    def test_unavailable_isolation_is_declined(self):
        job = self.run_job(isolation="kvm-microvm")
        self.assert_matches_reference(job)
        self.assertEqual(job["metrics"]["policy_rejections"], 8)
        self.assertEqual(job["metrics"]["accepted"], 0)
        self.assertEqual(job["metrics"]["recovered"], 8)

    def test_control_origin_and_worker_authentication(self):
        with self.assertRaises(urllib.error.HTTPError) as response:
            self.api("/api/jobs", {"steps":4}, {"Origin":"https://untrusted.example"})
        self.assertEqual(response.exception.code, 403)
        response.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as response:
            self.api("/api/worker/poll", {"worker_id":"volunteer-a"})
        self.assertEqual(response.exception.code, 401)
        response.exception.close()

    def test_cancellation_does_not_update_model_after_stop(self):
        self.api("/api/jobs", {"steps": 80})
        self.api("/api/cancel", {})
        before = self.api("/api/state")["job"]
        time.sleep(.6)
        after = self.api("/api/state")["job"]
        self.assertEqual(after["status"], "cancelled")
        self.assertEqual(after["weights"], before["weights"])
        self.assertEqual(after["history"], before["history"])

    def test_requester_and_one_inbound_volunteer(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        process = subprocess.Popen([sys.executable, "-m", "mtcp", "--demo", "--nodes", "1",
            "--port", str(port), "--database", str(Path(self.temp.name)/"single.db")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    if len(request(base, "/api/state")["workers"]) == 1:
                        break
                except OSError:
                    pass
                time.sleep(.1)
            else:
                self.fail("Single volunteer did not connect")
            request(base, "/api/jobs", body={"steps":4})
            for _ in range(100):
                job = request(base, "/api/state")["job"]
                if job["status"] != "running":
                    break
                time.sleep(.1)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["partitions"], [[400,1000]])
            self.assertEqual(job["metrics"]["accepted"], 4)
            self.assert_matches_reference(job)
        finally:
            process.terminate()
            process.wait(timeout=8)


if __name__ == "__main__":
    unittest.main()
