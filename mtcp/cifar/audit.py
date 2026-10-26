"""Independent replay comparison with bounded, persisted numerical evidence."""
import numpy as np
from .training import decode

ATOL = 1e-6
RTOL = 1e-5


def compare_update(reference, submitted):
    evidence = {'passed': True, 'atol': ATOL, 'rtol': RTOL}
    for name in ('weights', 'optimizer_momentum'):
        expected = decode(reference[name]).astype(np.float64)
        actual = decode(submitted[name]).astype(np.float64)
        difference = np.abs(expected - actual)
        violations = difference > ATOL + RTOL * np.abs(expected)
        evidence[name] = dict(
            passed=not bool(violations.any()),
            max_abs=float(difference.max()),
            mean_abs=float(difference.mean()),
            relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-30)),
            mismatched_coordinates=int(violations.sum()),
        )
        evidence['passed'] &= evidence[name]['passed']
    return evidence
