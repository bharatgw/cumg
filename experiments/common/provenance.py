"""Versioned experiment sidecars, serialized outside solver timing regions."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.common.paths import REPO


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def array_hash(array) -> dict:
    value = np.ascontiguousarray(array)
    description = {"dtype": value.dtype.str, "shape": list(value.shape), "order": "C"}
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode())
    digest.update(value.tobytes(order="C"))
    return {**description, "sha256": digest.hexdigest()}


def environment() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "jax", "jaxlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = {}
    for label, arguments in (("commit", ["rev-parse", "HEAD"]), ("dirty", ["status", "--porcelain"])):
        result = subprocess.run(["git", *arguments], cwd=REPO, capture_output=True, text=True)
        git[label] = (
            (bool(result.stdout.strip()) if label == "dirty" else result.stdout.strip())
            if result.returncode == 0
            else None
        )
    sources = sorted((REPO / "src/cumg").glob("*.py")) + sorted((REPO / "experiments/common").glob("*.py"))
    sources += sorted((REPO / "experiments/runners").glob("*.py"))
    return {
        "python": sys.version,
        "packages": packages,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "git": git,
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }


def save_profile(path: Path, *, args, row: dict, method: str, result, profile, arrays) -> None:
    """Save only the actual returned profile; initial profiles are never substituted."""
    certificate = getattr(result, "certificate", None)
    if certificate is None:
        certificate = getattr(result, "extra", {}).get("certificate")
    metadata = getattr(result, "metadata", None)
    if certificate is None and isinstance(metadata, dict):
        candidate = (
            metadata
            if getattr(result, "success", False)
            else metadata.get("best_candidate") or metadata.get("best_regret") or {}
        )
        support = candidate.get("support_certificate", {})
        certificate = candidate.get("certificate", support.get("certificate"))
    effective = dict(vars(args))
    if not method.startswith("stochastic_") and method not in ("full_batch", "minibatch"):
        for key in (
            "entropy_kappa",
            "smoothing_tau",
            "step_size",
            "step_decay",
            "logit_bound",
            "max_iter",
            "stochastic_n_random_starts",
            "certify_every",
            "stochastic_regret_tolerance",
            "jit_updates",
            "stagnation_window",
            "stagnation_rtol",
            "stagnation_atol",
            "gradient_clip_norm",
            "batch_size",
        ):
            effective[key] = None
    record = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "task": {key: row[key] for key in ("risk", "K", "n", "seed")},
        "configuration": effective,
        "inputs": {key: array_hash(value) for key, value in zip(("A", "B", "p"), arrays, strict=True)},
        "environment": environment(),
        "x": profile[0],
        "y": profile[1],
        "theta": getattr(result, "theta", None),
        "certificate": certificate,
        "selected_start": getattr(result, "selected_start", None),
        "selected_checkpoint": getattr(result, "best_certificate", None),
        "history": getattr(result, "history", None),
        "metrics": {
            key.removeprefix(method + "_"): value for key, value in row.items() if key.startswith(method + "_")
        },
        "lp_options": {
            "library": "scipy.optimize.linprog",
            "method": "highs",
            "options": None,
            "tolerances": "Defaults of the recorded SciPy/HiGHS version; no explicit overrides",
            "status": "Success and message preserved in certificate best responses where returned",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
