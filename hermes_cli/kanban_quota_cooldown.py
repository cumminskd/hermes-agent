"""Small, non-secret worker-to-dispatcher quota reset contract."""
import json
import math
import os
import sys

MARKER = "[kanban-quota-reset] "


def reset_epoch(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def emit_reset(result):
    """Emit only PID and typed reset metadata, never provider error text."""
    payload = {"pid": os.getpid(), "run_id": os.environ.get("HERMES_KANBAN_RUN_ID"),
               "quota_exhausted": result.get("failure_reason") not in {"overloaded", "server_error", "timeout"},
               "resets_at": reset_epoch(result.get("failure_resets_at"))}
    print(MARKER + json.dumps(payload), file=sys.stderr, flush=True)


def read_reset(log, pid, run_id=None):
    # Ignore earlier workers in append-mode logs, including stale reset times.
    for line in reversed((log or "").splitlines()):
        if not line.startswith(MARKER):
            continue
        try:
            item = json.loads(line[len(MARKER):])
        except (ValueError, TypeError):
            continue
        if (isinstance(item, dict) and type(item.get("pid")) is int
                and item["pid"] == pid and run_id is not None
                and item.get("run_id") == str(run_id)
                and isinstance(item.get("quota_exhausted"), bool)):
            return {"quota_exhausted": item["quota_exhausted"],
                    "quota_resets_at": reset_epoch(item.get("resets_at"))}
    return {"quota_exhausted": True, "quota_resets_at": None}
