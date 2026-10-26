"""Run python3 -m mtcp --demo, or python3 -m mtcp worker --help."""
import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .coordinator import Conflict, Coordinator

WEB = Path(__file__).resolve().parent.parent / "web"


def handler_for(coordinator):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, value, code=200):
            self.send_bytes(json.dumps(value, allow_nan=False).encode(), "application/json", code)

        def send_bytes(self, body, content_type, code=200, attachment=False):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
            if attachment:
                self.send_header("Content-Disposition", 'attachment; filename="mtcp-run.json"')
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            # Dashboard control is loopback-only, including when worker transport binds a LAN IP.
            host = self.headers.get("Host", "")
            if host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
                return False
            origin = self.headers.get("Origin")
            return self.client_address[0] == "127.0.0.1" and (not origin or origin == f"http://{host}")

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/health":
                self.send({"status": "ok"})
                return
            if route == "/v1/resources":
                self.send(dict(coordinator.requester_resources, role="requester", worker_id="requester"))
                return
            if route == "/v1/nodes":
                with coordinator.lock:
                    nodes = [dict(id=w["id"], endpoint=w.get("endpoint"), resources=w.get("resources"),
                                  online=time.time() - w["last_seen"] < 5)
                             for w in coordinator.workers.values()]
                self.send({"requester": coordinator.requester_resources, "volunteers": nodes})
                return
            if not self.local_request():
                self.send({"error": "Dashboard is available on localhost only"}, 403)
                return
            if route == "/api/state":
                self.send(coordinator.state())
            elif route == "/api/export":
                with coordinator.lock:
                    exported = dict(coordinator.state(), jobs=list(coordinator.jobs.values()))
                    body = json.dumps(exported, indent=2).encode()
                self.send_bytes(body,
                                "application/json", attachment=True)
            elif route in ("/", "/app.js", "/style.css"):
                name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[route]
                mime = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript", "style.css": "text/css"}[name]
                self.send_bytes((WEB / name).read_bytes(), mime)
            else:
                self.send({"error": "Not found"}, 404)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Request body must be between 1 and 16,384 bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                route = urlsplit(self.path).path
                if route.startswith("/api/worker/"):
                    worker_id = body.get("worker_id")
                    token = coordinator.tokens.get(worker_id)
                    if not token or not secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                        self.send({"error": "Invalid worker credentials"}, 401)
                        return
                    if route == "/api/worker/register":
                        coordinator.register(worker_id, body["policy"], body.get("resources"), body.get("endpoint"))
                        self.send({"status": "registered"})
                    elif route == "/api/worker/poll":
                        self.send(coordinator.poll(worker_id))
                    elif route == "/api/worker/result":
                        self.send(coordinator.submit(worker_id, body["result"], body["mac"]))
                    else:
                        self.send({"error": "Not found"}, 404)
                elif not self.local_request():
                    self.send({"error": "Run controls require a same-origin localhost request"}, 403)
                elif route == "/api/jobs":
                    self.send({"id": coordinator.start(body)}, 201)
                elif route == "/api/cancel":
                    coordinator.cancel()
                    self.send({"status": "cancelled"})
                else:
                    self.send({"error": "Not found"}, 404)
            except Conflict as exc:
                self.send({"error": str(exc)}, 409)
            except (ValueError, KeyError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)
    return Handler


def main():
    if len(sys.argv)>1 and sys.argv[1]=="peer":
        from .cifar.__main__ import main as cifar_main
        return cifar_main()
    parser = argparse.ArgumentParser(description="MTCP fixed-workload research prototype")
    parser.add_argument("command", nargs="?", choices=["serve", "worker"], default="serve")
    parser.add_argument("--demo", action="store_true", help="Launch two separate volunteer processes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--database", default=".mtcp/state.sqlite3")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--id", default="volunteer-a")
    parser.add_argument("--backend", choices=["bounded-process", "kvm-microvm"], default="bounded-process")
    parser.add_argument("--assets", default=".mtcp/firecracker")
    parser.add_argument("--nodes", type=int, default=2, help="Number of local demo volunteers (1–8)")
    parser.add_argument("--listen-host", default="127.0.0.1", help="Volunteer inbound address; use 0.0.0.0 for LAN")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument("--advertise-port", type=int, help="Routable volunteer port, if different through a tunnel")
    parser.add_argument("--cpus", help="Comma-separated allowed host CPU IDs")
    parser.add_argument("--memory-mb", type=int, default=128)
    parser.add_argument("--upload-mbps", type=float, default=100)
    parser.add_argument("--download-mbps", type=float, default=100)
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    cpus = [int(c) for c in args.cpus.split(",")] if args.cpus else allowed[:1]
    if not cpus or len(set(cpus)) != len(cpus) or any(c not in allowed for c in cpus) or len(cpus) > 32:
        parser.error("--cpus must select distinct available CPU IDs (at most 32)")
    if not 128 <= args.memory_mb <= 4096 or not 0 < args.upload_mbps <= 1000000 or not 0 < args.download_mbps <= 1000000:
        parser.error("Invalid memory or bandwidth budget")
    if not 1 <= args.nodes <= 8:
        parser.error("--nodes must be between 1 and 8")
    if args.advertise_port is not None and not 1 <= args.advertise_port <= 65535:
        parser.error("--advertise-port must be between 1 and 65535")
    if args.command == "worker":
        from .worker import work
        token = os.environ.get("MTCP_WORKER_TOKEN")
        if not token:
            parser.error("Set MTCP_WORKER_TOKEN to this worker's pre-shared key")
        os.sched_setaffinity(0, cpus)
        work(args.url.rstrip("/"), args.id, token, host=args.listen_host, port=args.listen_port,
             advertise_host=args.advertise_host, advertise_port=args.advertise_port,
             backend=args.backend, assets=args.assets, cpus=cpus,
             memory_mb=args.memory_mb, upload_mbps=args.upload_mbps, download_mbps=args.download_mbps)
        return
    tokens = json.loads(os.environ.get("MTCP_WORKER_TOKENS", "{}"))
    if args.demo:
        tokens = {f"volunteer-{chr(97+i)}": secrets.token_hex(32) for i in range(args.nodes)}
    if not isinstance(tokens, dict) or any(not isinstance(k, str) or not isinstance(v, str) or len(v) < 32 for k, v in tokens.items()):
        parser.error("MTCP_WORKER_TOKENS must map worker IDs to keys of at least 32 characters")
    Path(args.database).parent.mkdir(parents=True, exist_ok=True)
    coordinator = Coordinator(args.database, tokens)
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler_for(coordinator))
    except OSError as exc:
        parser.exit(2, f"Cannot start coordinator on {args.host}:{args.port}: {exc}. Choose another --port.\n")
    children = []
    if args.demo:
        for index, (worker_id, token) in enumerate(tokens.items()):
            address = f"127.0.0.{index+2}"
            node_cpus = [allowed[(index * 2 + offset) % len(allowed)]
                         for offset in range(min(1 + index % 2, len(allowed)))]
            children.append(subprocess.Popen([sys.executable, "-m", "mtcp", "worker", "--id", worker_id,
                "--url", f"http://127.0.0.1:{server.server_port}", "--listen-host", address,
                "--advertise-host", address, "--backend", args.backend, "--assets", args.assets,
                "--cpus", ",".join(map(str, node_cpus)), "--memory-mb", str(args.memory_mb),
                "--upload-mbps", str(25 * (index+1)), "--download-mbps", str(50 * (index+1))],
                env=dict(os.environ, MTCP_WORKER_TOKEN=token)))
    print(f"MTCP dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.stopped.set()
        for child in children:
            child.terminate()
        for child in children:
            child.wait(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
