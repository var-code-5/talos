import copy
import json
import math
import secrets
import sqlite3
import threading
import time
import uuid

from .kernel import digest, gradient, loss, samples, valid_gradient
from .protocol import ARTIFACT, BINDINGS, authenticated, sign
from .resources import advertise, validate_resources
from .transport import request
from urllib.parse import urlsplit


class Conflict(ValueError):
    pass


class Coordinator:
    def __init__(self, database, tokens):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, snapshot TEXT)")
        self.db.commit()
        self.jobs = {}
        for identifier, snapshot in self.db.execute("SELECT id, snapshot FROM jobs"):
            job = json.loads(snapshot)
            if job["status"] == "running":
                job["status"] = "interrupted"
            self.jobs[identifier] = job
        self.tokens = tokens
        self.workers = {}
        self.assignments = {}
        self.stopped = threading.Event()
        self.requester_resources = advertise()

    def save(self, job):
        self.db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?)", (job["id"], json.dumps(job)))
        self.db.commit()

    def event(self, job, kind, message, **extra):
        job["events"].append(dict(time=time.time(), kind=kind, message=message, **extra))

    def register(self, worker_id, policy, resources=None, endpoint=None):
        if worker_id not in self.tokens:
            raise ValueError("Unknown worker")
        if (not isinstance(policy, dict) or policy.get("isolation") not in ("bounded-process", "kvm-microvm")
                or type(policy.get("memory_mb")) is not int or not 128 <= policy["memory_mb"] <= 4096
                or policy.get("cpu_seconds") != 2):
            raise ValueError("Unsupported worker policy")
        if resources is not None:
            validate_resources(resources)
            if resources.get("isolation") != policy["isolation"]:
                raise ValueError("Capability and policy isolation mismatch")
            if resources["offered_memory_mb"] != policy["memory_mb"] or resources["offered_vcpus"] != policy.get("vcpus", 1):
                raise ValueError("Resource advertisement disagrees with owner policy")
        if endpoint is not None:
            parsed = urlsplit(endpoint)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
                raise ValueError("Endpoint must be an HTTP(S) origin")
        with self.lock:
            prior = self.workers.get(worker_id, {})
            need_probe = endpoint and (prior.get("endpoint") != endpoint or time.time() - prior.get("connection_checked_at", 0) > 30)
        connection = self.probe(worker_id, endpoint) if need_probe else None
        with self.lock:
            prior = self.workers.get(worker_id, {})
            self.workers[worker_id] = dict(id=worker_id, policy=policy, last_seen=time.time(),
                audited=prior.get("audited", 0), failures=prior.get("failures", 0),
                accepted=prior.get("accepted", 0),
                status=("computing" if resources.get("busy") else "idle") if resources else prior.get("status", "idle"),
                endpoint=endpoint, resources=resources or prior.get("resources"),
                connection=connection or prior.get("connection"),
                connection_checked_at=time.time() if connection else prior.get("connection_checked_at", 0))

    def probe(self, worker_id, endpoint):
        key = self.tokens[worker_id]
        def once(padding=0, download=0):
            nonce = secrets.token_hex(16)
            before = time.perf_counter()
            response = request(endpoint, "/v1/probe", key,
                dict(nonce=nonce, padding="x" * padding, download_bytes=download))
            elapsed = time.perf_counter() - before
            proof = response["proof"]
            if not authenticated(proof, response["mac"], key) or proof.get("nonce") != nonce or proof.get("worker_id") != worker_id:
                raise ValueError("Volunteer endpoint identity verification failed")
            if len(proof.get("padding", "")) != download:
                raise ValueError("Incomplete bandwidth probe")
            return elapsed
        rtt = once()
        download = once(download=131072)
        upload = once(padding=131072)
        return dict(reachable=True, rtt_ms=round(rtt * 1000, 2),
                    node_upload_mbps=round(131072 * 8 / download / 1e6, 2),
                    node_download_mbps=round(131072 * 8 / upload / 1e6, 2),
                    method="128 KiB end-to-end HTTP transfer, including request overhead", measured_at=time.time())

    def start(self, options):
        steps = options.get("steps", 24)
        if type(steps) is not int or not 4 <= steps <= 80:
            raise ValueError("Steps must be an integer between 4 and 80")
        audit = options.get("audit", "full")
        scenario = options.get("scenario", "honest")
        isolation = options.get("isolation", "bounded-process")
        if audit not in ("full", "adaptive") or scenario not in ("honest", "corrupt", "timeout", "replay"):
            raise ValueError("Unknown audit policy or scenario")
        if isolation not in ("bounded-process", "kvm-microvm"):
            raise ValueError("Unknown isolation requirement")
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise Conflict("A training run is already active")
            available = [w["id"] for w in self.workers.values() if time.time() - w["last_seen"] < 5
                         and (w.get("resources") or {}).get("accepting", True)]
            if not available:
                raise Conflict("At least one accepting volunteer must be online")
            available = sorted(available)
            if len(available) > 32:
                raise Conflict("At most 32 volunteers are supported in this prototype")
            deadline = options.get("task_timeout", 12 if isolation == "kvm-microvm" else 3)
            if type(deadline) not in (int, float) or not math.isfinite(deadline) or not 1 <= deadline <= 60:
                raise ValueError("Task timeout must be between 1 and 60 seconds")
            # Offered vCPU budgets determine disjoint remote partition sizes.
            capacities = [(self.workers[w].get("resources") or {}).get("offered_vcpus", 1) for w in available]
            total = sum(capacities)
            boundaries = [400]
            cumulative = 0
            for capacity in capacities:
                cumulative += capacity
                boundaries.append(400 + round(600 * cumulative / total))
            if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
                raise ValueError("CPU allocations produce an empty training partition")
            partitions = list(zip(boundaries, boundaries[1:]))
            identifier = uuid.uuid4().hex[:12]
            job = dict(id=identifier, status="running", created_at=time.time(), steps=steps,
                step=0, audit=audit, scenario=scenario, isolation=isolation, weights=[0., 0., 0.],
                history=[dict(step=0, loss=loss([0., 0., 0.], samples()))],
                tasks=[], events=[], workers=available, partitions=partitions, task_timeout=deadline,
                metrics=dict(accepted=0, audited=0, unaudited=0, rejected=0, recovered=0,
                             timeouts=0, policy_rejections=0, compute_ms=0., audit_ms=0.))
            self.jobs[identifier] = job
            self.event(job, "started", "Training started: 1,000 public synthetic samples per step")
            self.save(job)
            threading.Thread(target=self.run, args=(identifier,), daemon=True).start()
            return identifier

    def audit_probability(self, worker, mode):
        if mode == "full" or worker["failures"]:
            return 1.
        return .15 if worker["audited"] >= 5 else .60

    def create_task(self, job, worker_id, start, end):
        payload = dict(seed=42, start=start, end=end, weights=job["weights"][:])
        contract = dict(job_id=job["id"], task_id=f'{job["step"]}:{start}', attempt=1,
            model_version=job["step"] - 1, input_digest=digest(payload), artifact_digest=ARTIFACT,
            expires_at=time.time() + job.get("task_timeout", 3), isolation=job["isolation"], memory_mb=128,
            cpu_seconds=2, data_classification="public", payload=payload)
        contract["vcpus"] = (self.workers[worker_id].get("resources") or {}).get("offered_vcpus", 1)
        task = dict(contract=contract, signature=sign(contract, self.tokens[worker_id]),
                    worker_id=worker_id, status="assigned", audit=None, result=None,
                    reason=None, duration_ms=0.)
        self.assignments[(job["id"], contract["task_id"])] = task
        job["tasks"].append(task)
        return task

    def dispatch(self, job, task):
        worker_id = task["worker_id"]
        endpoint = self.workers[worker_id].get("endpoint")
        if not endpoint:
            return  # Compatibility with legacy pull controllers.
        assignment = dict(contract=task["contract"], signature=task["signature"],
            fault=job["scenario"] if worker_id == job["workers"][-1] and job["step"] == 3 else "honest")
        with self.lock:
            task["status"] = "running"
            self.workers[worker_id]["status"] = "computing"
        while not self.stopped.is_set() and time.time() < task["contract"]["expires_at"]:
            try:
                request(endpoint, "/v1/assignments", self.tokens[worker_id], assignment, timeout=2)
                return
            except OSError as exc:
                # Busy is transient (e.g. a previous expired computation is tearing down).
                if getattr(exc, "code", 0) not in (0, 409):
                    break
                self.stopped.wait(.1)
        with self.lock:
            if task["status"] == "running":
                self.event(job, "connection", f"{worker_id}: direct dispatch could not be acknowledged")

    def poll(self, worker_id):
        with self.lock:
            worker = self.workers.get(worker_id)
            if worker is None:
                raise Conflict("Register the worker first")
            worker["last_seen"] = time.time()
            for job in self.jobs.values():
                if job["status"] != "running":
                    continue
                for task in job["tasks"]:
                    if task["worker_id"] == worker_id and task["status"] == "assigned":
                        task["status"] = "running"
                        worker["status"] = "computing"
                        # Faults are deliberate experiment instructions, outside the security contract.
                        fault = job["scenario"] if worker_id == job["workers"][-1] and job["step"] == 3 else "honest"
                        return dict(contract=task["contract"], signature=task["signature"], fault=fault)
            worker["status"] = "idle"
            return None

    def submit(self, worker_id, result, mac):
        if not authenticated(result, mac, self.tokens[worker_id]):
            raise ValueError("Result authentication failed")
        with self.lock:
            task = self.assignments.get((result.get("job_id"), result.get("task_id")))
            if task is None or task["worker_id"] != worker_id:
                raise Conflict("Unknown assignment")
            job = self.jobs[result["job_id"]]
            if job["status"] != "running" or task["status"] not in ("assigned", "running"):
                raise Conflict("Result is stale or already committed")
            contract = task["contract"]
            reason = None
            if time.time() > contract["expires_at"]:
                reason = "Assignment expired"
            elif any(result.get(key) != contract[key] for key in BINDINGS):
                reason = "Assignment metadata mismatch / replay"
            elif "declined" in result:
                reason = str(result["declined"])[:160]
                job["metrics"]["policy_rejections"] += 1
            elif not valid_gradient(result.get("gradient")):
                reason = "Invalid gradient structure or non-finite values"
            # The whole submission is committed before drawing the private audit coin.
            task["result"] = copy.deepcopy(result)
            task["status"] = "committed"
            worker = self.workers[worker_id]
            worker["status"] = "idle"
            if reason is None:
                probability = self.audit_probability(worker, job["audit"])
                task["audit_probability"] = probability
                task["audit"] = secrets.randbelow(10**9) < int(probability * 10**9)
                if task["audit"]:
                    before = time.perf_counter()
                    p = contract["payload"]
                    expected = gradient(p["weights"], samples(p["seed"])[p["start"]:p["end"]])
                    job["metrics"]["audit_ms"] += (time.perf_counter() - before) * 1000
                    job["metrics"]["audited"] += 1
                    if not all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-9)
                               for a, b in zip(expected, result["gradient"])):
                        reason = "Independent recomputation detected a corrupted gradient"
                    else:
                        worker["audited"] += 1
                else:
                    job["metrics"]["unaudited"] += 1
            if reason:
                task["status"] = "rejected"
                task["reason"] = reason
                job["metrics"]["rejected"] += 1
                if "declined" not in result:
                    worker["failures"] += 1
                self.event(job, "rejected", f"{worker_id}: {reason}", task_id=contract["task_id"])
            else:
                task["status"] = "accepted"
                worker["accepted"] += 1
                job["metrics"]["accepted"] += 1
                duration = result.get("duration_ms", 0.)
                if type(duration) in (int, float) and math.isfinite(duration) and 0 <= duration <= 10000:
                    task["duration_ms"] = duration
                    job["metrics"]["compute_ms"] += duration
                self.event(job, "audited" if task["audit"] else "accepted",
                    f'{worker_id}: gradient {"independently audited" if task["audit"] else "accepted without audit"}',
                    task_id=contract["task_id"])
            self.save(job)
            return {"status": task["status"], "audited": task["audit"]}

    def run(self, identifier):
        try:
            rows = samples()
            for step in range(1, self.jobs[identifier]["steps"] + 1):
                with self.lock:
                    job = self.jobs[identifier]
                    if self.stopped.is_set() or job["status"] != "running":
                        return
                    job["step"] = step
                    weights = job["weights"][:]
                    tasks = [self.create_task(job, job["workers"][i], start, end)
                             for i, (start, end) in enumerate(job["partitions"])]
                    self.save(job)
                for task in tasks:
                    threading.Thread(target=self.dispatch, args=(job, task), daemon=True).start()
                local = gradient(weights, rows[:400])
                while not self.stopped.wait(.04):
                    with self.lock:
                        if job["status"] != "running":
                            return
                        for task in tasks:
                            if task["status"] in ("assigned", "running") and time.time() > task["contract"]["expires_at"]:
                                task["status"] = "timed_out"
                                task["reason"] = "Worker missed the assignment deadline"
                                job["metrics"]["timeouts"] += 1
                                self.event(job, "timeout", f'{task["worker_id"]}: deadline missed')
                        if all(t["status"] in ("accepted", "rejected", "timed_out") for t in tasks):
                            break
                if self.stopped.is_set():
                    return
                with self.lock:
                    if job["status"] != "running":
                        return
                    contributions = [(400, local)]
                    for task in tasks:
                        p = task["contract"]["payload"]
                        if task["status"] == "accepted":
                            contribution = task["result"]["gradient"]
                        else:
                            # Bounded retry: new attempt on trusted requester, same logical work unit.
                            contribution = gradient(weights, rows[p["start"]:p["end"]])
                            task["recovery"] = dict(attempt=2, worker_id="requester", status="accepted")
                            job["metrics"]["recovered"] += 1
                            self.event(job, "recovered", f'{task["worker_id"]}: partition recomputed on requester')
                        contributions.append((p["end"] - p["start"], contribution))
                    job["weights"] = [weights[i] - .3 * sum(n * g[i] for n, g in contributions) / 1000
                                      for i in range(3)]
                    for task in tasks:
                        task["aggregated"] = True
                    job["history"].append(dict(step=step, loss=loss(job["weights"], rows)))
                    self.event(job, "aggregated", f"Model v{step}: all 1,000 samples aggregated exactly once")
                    self.save(job)
                self.stopped.wait(.15)
            with self.lock:
                if job["status"] != "running" or self.stopped.is_set():
                    return
                job["status"] = "completed"
                job["finished_at"] = time.time()
                self.event(job, "completed", "Training completed; checkpoint saved")
                self.save(job)
        except Exception as exc:
            with self.lock:
                job = self.jobs[identifier]
                job["status"] = "failed"
                self.event(job, "error", str(exc))
                self.save(job)

    def cancel(self):
        with self.lock:
            for job in self.jobs.values():
                if job["status"] == "running":
                    job["status"] = "cancelled"
                    self.event(job, "cancelled", "Run stopped by requester")
                    self.save(job)

    def state(self):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)
            workers = [dict(w, online=time.time() - w["last_seen"] < 5) for w in self.workers.values()]
            return copy.deepcopy(dict(workers=workers, requester_resources=self.requester_resources,
                job=jobs[0] if jobs else None,
                runs=[{k: j[k] for k in ("id", "status", "created_at", "scenario", "audit", "steps")}
                      for j in jobs], artifact_digest=ARTIFACT))
