"""Deterministic fingerprints for saved portfolio-run checkpoints.

The capacity and dynamic-NAV runners can resume expensive completed runs.  A
checkpoint is reusable only when the economic inputs that produced it are the
same.  This module hashes the signal panel, economically relevant configuration,
and portfolio keyword arguments so stale outputs are not silently reused after a
configuration or input change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional
import hashlib
import json

import numpy as np
import pandas as pd


RUN_SIGNATURE_VERSION = "cost-aware-portfolio-v1"


_IMPLEMENTATION_FILES = (
    "optimizer.py",
    "covariance.py",
    "state.py",
    "transaction_costs.py",
    "strategies.py",
)


def portfolio_implementation_fingerprint() -> str:
    """Hash the portfolio source files that determine economic run behavior.

    Checkpoint validity should change when the implementation changes, even if a
    caller forgets to change an output directory or manually bump a version tag.
    The files are small relative to the data and are hashed once per run request.
    """
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for filename in _IMPLEMENTATION_FILES:
        path = root / filename
        if not path.exists():
            raise RuntimeError("Portfolio implementation file is missing: %s" % path)
        h.update(filename.encode("utf-8"))
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Return a deterministic content fingerprint for a pandas DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    h = hashlib.sha256()
    h.update(str(frame.shape).encode("utf-8"))
    h.update(json.dumps([str(c) for c in frame.columns], separators=(",", ":")).encode("utf-8"))
    h.update(json.dumps([str(x) for x in frame.dtypes], separators=(",", ":")).encode("utf-8"))
    # pandas' row hash is stable for the same values/index and avoids materializing
    # a huge CSV representation for large signal panels.
    row_hash = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    h.update(np.asarray(row_hash.to_numpy(dtype="uint64"), dtype="uint64").tobytes())
    return h.hexdigest()


def _series_fingerprint(series: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(str(series.name).encode("utf-8"))
    h.update(str(series.dtype).encode("utf-8"))
    row_hash = pd.util.hash_pandas_object(series, index=True, categorize=True)
    h.update(np.asarray(row_hash.to_numpy(dtype="uint64"), dtype="uint64").tobytes())
    return h.hexdigest()


def _path_identity(value: Any) -> Mapping[str, Any]:
    p = Path(value).expanduser()
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    out = {"path": str(resolved)}
    if resolved.exists() and resolved.is_file():
        stat = resolved.stat()
        out.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return out


def _canonicalize(value: Any, *, key_hint: Optional[str] = None) -> Any:
    """Convert nested inputs into JSON-serializable deterministic objects."""
    if value is None or isinstance(value, (bool, str, int)):
        if isinstance(value, str) and key_hint and key_hint.endswith("_path"):
            return {"__file__": _path_identity(value)}
        return value
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if np.isnan(x):
            return "NaN"
        if np.isposinf(x):
            return "+Inf"
        if np.isneginf(x):
            return "-Inf"
        return x
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return {"__file__": _path_identity(value)}
    if isinstance(value, pd.DataFrame):
        return {"__dataframe_sha256__": dataframe_fingerprint(value)}
    if isinstance(value, pd.Series):
        return {"__series_sha256__": _series_fingerprint(value)}
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        h = hashlib.sha256()
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(np.ascontiguousarray(arr).tobytes())
        return {"__ndarray_sha256__": h.hexdigest()}
    if isinstance(value, Mapping):
        return {
            str(k): _canonicalize(v, key_hint=str(k))
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted((_canonicalize(v) for v in value), key=lambda x: repr(x))
    # Configuration values should be simple.  A stable type-qualified repr keeps
    # unexpected custom values explicit rather than silently dropping them.
    return {"__repr__": "%s.%s:%r" % (type(value).__module__, type(value).__qualname__, value)}


def build_run_signature(
    *,
    run_kind: str,
    model: str,
    signal_df: pd.DataFrame,
    config_fields: Mapping[str, Any],
    portfolio_kwargs: Optional[Mapping[str, Any]] = None,
    manager_identity: Optional[str] = None,
) -> str:
    """Return a SHA-256 signature for one resumable portfolio run."""
    payload = {
        "signature_version": RUN_SIGNATURE_VERSION,
        "run_kind": str(run_kind),
        "model": str(model),
        "signal_sha256": dataframe_fingerprint(signal_df),
        "config": _canonicalize(dict(config_fields)),
        "portfolio_kwargs": _canonicalize(dict(portfolio_kwargs or {})),
        "manager_identity": None if manager_identity is None else str(manager_identity),
        "implementation_sha256": portfolio_implementation_fingerprint(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def manager_identity(manager_cls: Optional[Any]) -> str:
    """Return a stable class identity for checkpoint validation."""
    if manager_cls is None:
        return "unconfigured-manager"
    return "%s.%s" % (getattr(manager_cls, "__module__", ""), getattr(manager_cls, "__qualname__", repr(manager_cls)))


__all__ = [
    "RUN_SIGNATURE_VERSION",
    "dataframe_fingerprint",
    "portfolio_implementation_fingerprint",
    "build_run_signature",
    "manager_identity",
]
