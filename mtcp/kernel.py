"""Fixed, deterministic training kernel. No uploaded code or pickle is accepted."""
import hashlib
import json
import math
import random
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def samples(seed=42):
    rng = random.Random(seed)
    return [(x := rng.uniform(-1, 1), z := rng.uniform(-1, 1),
             2.4 * x - 1.7 * z + 0.6 + rng.gauss(0, .04)) for _ in range(1000)]


def gradient(weights, rows):
    total = [0., 0., 0.]
    for x, z, y in rows:
        error = weights[0] * x + weights[1] * z + weights[2] - y
        for i, value in enumerate((x, z, 1)):
            total[i] += 2 * error * value / len(rows)
    return total


def loss(weights, rows):
    return sum((weights[0] * x + weights[1] * z + weights[2] - y) ** 2
               for x, z, y in rows) / len(rows)


def valid_gradient(value):
    return (isinstance(value, list) and len(value) == 3
            and all(type(x) in (float, int) and math.isfinite(x) and abs(x) < 100 for x in value))


if __name__ == "__main__":
    # Limits apply to this disposable fixed-kernel child, not the controller.
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    payload = json.load(sys.stdin)
    rows = samples(payload["seed"])[payload["start"]:payload["end"]]
    print(json.dumps(gradient(payload["weights"], rows)))
