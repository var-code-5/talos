"""Prototype pairwise MAC authentication; not asymmetric signatures/attestation."""
import hashlib
import hmac
import time
from pathlib import Path

from .kernel import canonical, digest

ARTIFACT = hashlib.sha256(Path(__file__).with_name("kernel.py").read_bytes()).hexdigest()
BINDINGS = ("job_id", "task_id", "attempt", "model_version", "input_digest", "artifact_digest")


def sign(value, key):
    return hmac.new(key.encode(), canonical(value), hashlib.sha256).hexdigest()


def authenticated(value, signature, key):
    return isinstance(signature, str) and hmac.compare_digest(sign(value, key), signature)


def admit(contract, signature, key, policy):
    if not authenticated(contract, signature, key):
        return "Contract authentication failed"
    if contract["expires_at"] <= time.time():
        return "Contract expired"
    if contract["artifact_digest"] != ARTIFACT:
        return "Unknown execution artifact"
    if contract["isolation"] != policy["isolation"]:
        return "Required isolation is unavailable"
    if contract["memory_mb"] > policy["memory_mb"] or contract["cpu_seconds"] > policy["cpu_seconds"]:
        return "Resource request exceeds owner policy"
    if contract.get("vcpus", 1) > policy.get("vcpus", 1):
        return "CPU request exceeds owner policy"
    if contract["data_classification"] != "public":
        return "Owner permits public data only"
    payload = contract["payload"]
    if digest(payload) != contract["input_digest"]:
        return "Input digest mismatch"
    if not (0 <= payload["start"] < payload["end"] <= 1000):
        return "Invalid sample partition"
    return None
