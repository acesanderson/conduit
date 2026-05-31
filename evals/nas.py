"""NAS-backed eval artifact paths.

All eval jobs write results, status, and logs to $NAS/evals/<project>/<eval>/.
Filenames encode provenance: <eval>__<utc_ts>__<host>__<artifact>.<ext>.

$NAS must point at the NAS mount root on every host (universal env var).
If $NAS is unset or $NAS/evals/ is missing, jobs fail-fast at import.
NAS-backed storage is the standard; local fallback would silently fragment
results across hosts and defeat the point.
"""
from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from pathlib import Path


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _host() -> str:
    return socket.gethostname().split(".")[0]


def eval_artifact_dir(nas_project: str, eval_name: str) -> Path:
    """Return $NAS/evals/<nas_project>/<eval_name>/, creating subdirs if needed.

    Fail-fast if $NAS is unset or $NAS/evals/ is missing (NAS unmounted).
    """
    nas = os.environ.get("NAS")
    if not nas:
        raise SystemExit(
            "$NAS is not set — NAS-backed eval storage is required. "
            "Ensure /home/<user>/.exports (or ~/.zshrc on macOS) exports NAS."
        )
    root = Path(nas) / "evals"
    if not root.is_dir():
        raise SystemExit(
            f"$NAS/evals/ does not exist or NAS not mounted: {root}"
        )
    out = root / nas_project / eval_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def artifact_paths(nas_project: str, eval_name: str) -> dict[str, Path]:
    """Standard set of per-run artifact paths for a job.

    Returns a dict with `results_csv`, `status_json`, `log` keys.
    Each filename embeds the eval name, UTC timestamp, and hostname so that
    runs accumulate as history and two hosts writing the same eval do not
    collide.
    """
    d = eval_artifact_dir(nas_project, eval_name)
    prefix = f"{eval_name}__{_utc_ts()}__{_host()}"
    return {
        "results_csv": d / f"{prefix}__results.csv",
        "status_json": d / f"{prefix}__status.json",
        "log":         d / f"{prefix}__run.log",
    }
