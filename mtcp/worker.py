"""Volunteer server: public resource discovery and authenticated inbound assignments."""
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .protocol import BINDINGS, admit, sign
from .resources import advertise
from .transport import request

POLICY = dict(isolation='bounded-process', memory_mb=128, cpu_seconds=2)


class Volunteer:
    def __init__(self, worker_id, token, coordinator, backend='bounded-process', assets='.mtcp/firecracker',
                 cpus=None, memory_mb=128, upload_mbps=100, download_mbps=100):
        self.worker_id, self.token, self.coordinator = worker_id, token, coordinator
        self.cpus = cpus or sorted(os.sched_getaffinity(0))[:1]
        self.policy = dict(isolation=backend, memory_mb=memory_mb, cpu_seconds=2, vcpus=len(self.cpus))
        self.capacity = advertise(self.cpus, memory_mb, upload_mbps, download_mbps, backend)
        self.runner = None
        if backend == 'kvm-microvm':
            from .firecracker import Firecracker
            self.runner = Firecracker(assets, self.cpus, memory_mb)
            self.boot_probe = self.runner.execute(operation='probe')
        self.lock = threading.Lock()
        self.busy = False
        self.seen = set()
        self.previous = None
        self.stopped = threading.Event()

    def resources(self):
        with self.lock:
            return dict(self.capacity, accepting=True, available_slots=0 if self.busy else 1, busy=self.busy,
                        endpoint=self.endpoint, worker_id=self.worker_id, timestamp=time.time())

    def accept(self, assignment):
        contract = assignment['contract']
        reason = admit(contract, assignment['signature'], self.token, self.policy)
        identity = tuple(contract[key] for key in BINDINGS)
        with self.lock:
            if identity in self.seen:
                raise ValueError('Assignment already accepted')
            if self.busy:
                raise BlockingIOError('Volunteer has no available slot')
            self.seen.add(identity)
            self.busy = True
        threading.Thread(target=self.execute, args=(assignment, reason), daemon=True).start()
        return {'status': 'admitted' if reason is None else 'declined', 'reason': reason}

    def execute(self, assignment, reason):
        contract = assignment['contract']
        result = {key: contract[key] for key in BINDINGS}
        try:
            if reason:
                result['declined'] = reason
            else:
                start = time.perf_counter()
                remaining = max(.01, contract['expires_at'] - time.time())
                if self.runner:
                    execution = self.runner.execute(contract['payload'], timeout=min(8, remaining), memory_mb=contract['memory_mb'])
                    result['gradient'] = execution.pop('gradient')
                    result['execution'] = execution
                else:
                    kernel = Path(__file__).with_name('kernel.py')
                    child = subprocess.run([sys.executable, '-I', str(kernel)],
                        input=json.dumps(contract['payload']), capture_output=True, text=True,
                        timeout=min(2.5, remaining), cwd='/tmp', env={'PATH': os.defpath})
                    if child.returncode:
                        raise RuntimeError('Bounded training process failed')
                    result['gradient'] = json.loads(child.stdout)
                    result['execution'] = dict(isolation='bounded-process', fresh_vm=False)
                result['duration_ms'] = (time.perf_counter() - start) * 1000
                fault = assignment.get('fault', 'honest')
                if fault == 'corrupt':
                    result['gradient'][0] += 8.
                elif fault == 'replay' and self.previous:
                    result['model_version'] = self.previous['model_version']
                elif fault == 'timeout':
                    self.stopped.wait(max(0, contract['expires_at'] - time.time()) + .3)
                self.previous = result.copy()
                self.stopped.wait(.15)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            result['declined'] = f'Execution failed: {exc}'[:160]
        body = dict(worker_id=self.worker_id, result=result, mac=sign(result, self.token))
        try:
            for attempt in range(3):
                try:
                    request(self.coordinator, '/api/worker/result', self.token, body)
                    break
                except OSError as exc:
                    if getattr(exc, 'code', 0) in (400, 401, 409) or attempt == 2:
                        print(f'{self.worker_id}: result delivery: {exc}', flush=True)
                        break
                    self.stopped.wait(.2)
        finally:
            with self.lock:
                self.busy = False

    def heartbeat(self):
        while not self.stopped.is_set():
            try:
                request(self.coordinator, '/api/worker/register', self.token,
                    dict(worker_id=self.worker_id, policy=self.policy, endpoint=self.endpoint,
                         resources=self.resources()))
            except (OSError, ValueError) as exc:
                print(f'{self.worker_id}: heartbeat: {exc}', flush=True)
            self.stopped.wait(1)


def handler_for(node):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, value, status=200):
            raw = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            started = time.monotonic()
            for offset in range(0, len(raw), 16384):
                chunk = raw[offset:offset+16384]
                target = (offset + len(chunk)) * 8 / (node.capacity['offered_upload_mbps'] * 1e6)
                node.stopped.wait(max(0, target - (time.monotonic() - started)))
                self.wfile.write(chunk)

        def do_GET(self):
            if self.path == '/v1/resources':
                self.send(node.resources())
            elif self.path == '/health':
                self.send({'status':'ok', 'worker_id':node.worker_id})
            else:
                self.send({'error':'Not found'}, 404)

        def do_POST(self):
            self.connection.settimeout(5)
            if not secrets.compare_digest(self.headers.get('Authorization', ''), f'Bearer {node.token}'):
                self.send({'error':'Invalid requester credentials'}, 401)
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                limit = 300000 if self.path == '/v1/probe' else 16384
                if not 0 < length <= limit:
                    raise ValueError('Invalid request size')
                before = time.monotonic()
                raw = self.rfile.read(length)
                node.stopped.wait(max(0, len(raw)*8/(node.capacity['offered_download_mbps']*1e6) - (time.monotonic()-before)))
                value = json.loads(raw)
                if self.path == '/v1/probe':
                    size = value.get('download_bytes', 0)
                    if type(size) is not int or not 0 <= size <= 131072:
                        raise ValueError('Invalid probe size')
                    proof = dict(nonce=value['nonce'], worker_id=node.worker_id, padding='x' * size)
                    self.send(dict(proof=proof, mac=sign(proof, node.token)))
                elif self.path == '/v1/assignments':
                    self.send(node.accept(value), 202)
                else:
                    self.send({'error':'Not found'}, 404)
            except BlockingIOError as exc:
                self.send({'error':str(exc)}, 409)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.send({'error':str(exc)}, 400)
    return Handler


def work(url, worker_id, token, host='127.0.0.1', port=0, advertise_host='127.0.0.1', advertise_port=None, **options):
    node = Volunteer(worker_id, token, url, **options)
    server = ThreadingHTTPServer((host, port), handler_for(node))
    node.endpoint = f'http://{advertise_host}:{advertise_port or server.server_port}'
    threading.Thread(target=node.heartbeat, daemon=True).start()
    print(f'{worker_id} accepting connections at {node.endpoint}; isolation={node.policy["isolation"]}', flush=True)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.stopped.set()
        server.server_close()
