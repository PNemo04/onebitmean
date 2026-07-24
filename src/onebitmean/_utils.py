"""Internal validation, hashing, and immutable-array helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def readonly(array: np.ndarray, dtype: Any | None = None) -> np.ndarray:
    """Return a contiguous read-only array, optionally converted to ``dtype``."""
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def update_digest(digest: "hashlib._Hash", value: Any) -> None:
    """Add a stable representation of ``value`` to a hash digest."""
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    elif is_dataclass(value):
        digest.update(json.dumps(asdict(value), sort_keys=True).encode("utf-8"))
    else:
        digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))


def fingerprint(*values: Any) -> str:
    """Compute a SHA-256 fingerprint for a complete public query plan."""
    digest = hashlib.sha256()
    for value in values:
        update_digest(digest, value)
    return digest.hexdigest()


def validate_refinement_inputs(k: float, sigma: float, epsilon: float, tau: float) -> None:
    if not np.isfinite(k) or k <= 1:
        raise ValueError("k must be finite and strictly greater than one")
    for name, value in (("sigma", sigma), ("epsilon", epsilon), ("tau", tau)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if epsilon > tau:
        raise ValueError("the reference implementation requires epsilon <= tau")

