"""Opt-in real KVM tests. No mocked hypervisor or guest results."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from mtcp.firecracker import Firecracker
from mtcp.kernel import gradient, samples
from mtcp.transport import request


@unittest.skipUnless(os.environ.get('MTCP_TEST_FIRECRACKER') == '1', 'Set MTCP_TEST_FIRECRACKER=1 for real KVM tests')
class FirecrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cls.url = f'http://127.0.0.1:{cls.port}'
        cls.log = open(Path(cls.temp.name) / 'cluster.log', 'w+')
        cls.process = subprocess.Popen([sys.executable, '-m', 'mtcp', '--demo', '--nodes', '3',
            '--backend', 'kvm-microvm', '--port', str(cls.port),
            '--database', str(Path(cls.temp.name) / 'state.db')], stdout=cls.log, stderr=cls.log)
        for _ in range(200):
            try:
                state = request(cls.url, '/api/state')
                if len(state['workers']) == 3:
                    cls.runner = Firecracker()
                    return
            except OSError:
                pass
            time.sleep(.1)
        cls.process.terminate()
        cls.process.wait(timeout=8)
        cls.log.seek(0)
        raise RuntimeError(cls.log.read())

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.log.close()
        cls.temp.cleanup()

    def run_job(self, scenario):
        request(self.url, '/api/jobs', body=dict(steps=4, scenario=scenario, isolation='kvm-microvm', audit='full'))
        for _ in range(250):
            job = request(self.url, '/api/state')['job']
            if job['status'] != 'running':
                self.assertEqual(job['status'], 'completed', job['events'])
                return job
            time.sleep(.15)
        self.fail('Real microVM cluster did not complete')

    def reference(self, job):
        weights = [0., 0., 0.]
        for _ in range(4):
            g = gradient(weights, samples())
            weights = [w - .3*v for w,v in zip(weights,g)]
        for a,b in zip(weights, job['weights']):
            self.assertAlmostEqual(a,b,places=10)
        self.assertEqual(sum(b-a for a,b in job['partitions']), 600)
        self.assertEqual(len(job['workers']), 3)
        for task in job['tasks']:
            self.assertTrue(task['aggregated'])
            if task['status'] == 'accepted':
                self.assertEqual(task['result']['execution']['vmm'], 'firecracker')
                self.assertTrue(task['result']['execution']['fresh_vm'])

    def test_guest_isolation_and_limits(self):
        sentinel = Path(self.temp.name) / 'host-secret'
        sentinel.write_text('must never reach the guest')
        probe = self.runner.execute(operation='probe', sentinel=str(sentinel))
        self.assertEqual(probe['uid'], 65534)
        self.assertEqual(probe['interfaces'], ['lo'])
        self.assertFalse(probe['host_file_visible'])
        self.assertFalse(probe['rootfs_writable'])
        self.assertTrue(probe['rootfs_readonly'])
        self.assertEqual(probe['vcpus'], 1)
        self.assertLessEqual(probe['guest_memory_mb'], 128)
        self.assertFalse(probe['outbound_network'])
        self.assertTrue(probe['memory_limit_enforced'])
        self.assertNotEqual(probe['kernel'], os.uname().release)
        with self.assertRaisesRegex(RuntimeError, 'resource limit'):
            self.runner.execute(operation='cpu-probe', timeout=7)
        before = time.monotonic()
        with self.assertRaises((TimeoutError, socket.timeout)):
            self.runner.execute(operation='stall-probe', timeout=2)
        self.assertLess(time.monotonic()-before, 7)
        # A fresh VM can still run after forced teardown; no poisoned guest state is reused.
        self.assertIn('gradient', self.runner.execute(dict(seed=42,start=400,end=700,weights=[0,0,0])))

    def test_controller_crash_does_not_leave_guest_running(self):
        def active_scopes():
            result = subprocess.check_output(['systemctl', '--user', 'list-units',
                '--type=scope', '--state=active', '--plain', '--no-legend', 'mtcp-vm-*'], text=True)
            return {line.split()[0] for line in result.splitlines() if line.strip()}
        baseline = active_scopes()
        child = subprocess.Popen([sys.executable, '-c',
            "from mtcp.firecracker import Firecracker; Firecracker().execute(operation='stall-probe', timeout=20)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        created = set()
        try:
            for _ in range(100):
                created = active_scopes() - baseline
                if created:
                    break
                time.sleep(.1)
            self.assertEqual(len(created), 1, 'Expected one live VM scope before controller crash')
            # Allow the guest to boot and block in its task before killing its controller.
            time.sleep(1.5)
            self.assertIsNone(child.poll())
            child.kill()
            child.wait(timeout=5)
            for _ in range(40):
                if not (created & active_scopes()):
                    break
                time.sleep(.1)
            self.assertFalse(created & active_scopes(), 'Guest survived controller SIGKILL')
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            for unit in created:
                subprocess.run(['systemctl','--user','stop',unit], capture_output=True, timeout=5)

    def test_resource_advertisements_and_inbound_authentication(self):
        workers = request(self.url, '/api/state')['workers']
        self.assertEqual(len({w['endpoint'] for w in workers}), 3)
        self.assertEqual(len({w['resources']['host_id'] for w in workers}), 1)
        if len(os.sched_getaffinity(0)) >= 2:
            self.assertEqual(sorted(w['resources']['offered_vcpus'] for w in workers), [1,1,2])
        for worker in workers:
            resources = request(worker['endpoint'], '/v1/resources')
            self.assertTrue(resources['accepting'])
            self.assertEqual(resources['isolation'], 'kvm-microvm')
            self.assertGreater(resources['offered_vcpus'], 0)
            self.assertGreater(resources['gradient_samples_per_second'], 0)
            self.assertTrue(worker['connection']['reachable'])
            self.assertGreater(worker['connection']['node_upload_mbps'], 0)
            with self.assertRaises(urllib.error.HTTPError) as response:
                request(worker['endpoint'], '/v1/assignments', body={'contract':{}})
            self.assertEqual(response.exception.code, 401)
            response.exception.close()

    def test_three_volunteer_training_and_corruption_recovery(self):
        report = []
        for scenario in ('honest','corrupt','replay','timeout'):
            with self.subTest(scenario=scenario):
                job = self.run_job(scenario)
                self.reference(job)
                self.assertEqual(job['metrics']['recovered'], 0 if scenario == 'honest' else 1)
                self.assertEqual(job['metrics']['accepted'], 12 if scenario == 'honest' else 11)
                report.append({k:job[k] for k in ('id','scenario','status','weights','metrics','partitions')})
        Path('.mtcp/firecracker-test-report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    unittest.main()
