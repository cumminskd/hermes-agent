"""Tri-state worker identity checks shared by claim protection and signalling."""

import subprocess
from typing import Optional


def worker_identity_matches(pid: Optional[int], started_at, *, exact: bool = False) -> Optional[bool]:
    """True = match, False = proven conflict, None = identity unavailable.

    Shared start-time drift tolerance protects claims on macOS. Signals require
    exact equality: a close reading may be clock drift OR rapid PID reuse.
    Empty epoch (normal on macOS) supplies no boot evidence; never infer reuse
    merely because an epoch was unavailable on one of the observations.
    """
    from hermes_cli import kanban_db_dispatch as dispatch

    if not pid or started_at in (None, dispatch.UNVERIFIED_WORKER_FINGERPRINT):
        return None
    from gateway.status import get_process_start_time, start_time_fingerprints_match
    try:
        if isinstance(started_at, str) and "|" in started_at:
            current = dispatch._process_fingerprint(int(pid))
            if current is None:
                return None
            recorded_epoch, recorded_start = started_at.rsplit("|", 1)
            current_epoch, current_start = current.rsplit("|", 1)
        else:
            recorded_epoch = current_epoch = ""
            recorded_start = started_at
            current_start = get_process_start_time(int(pid))
        if current_start is None or int(recorded_start) <= 0 or int(current_start) <= 0:
            return None
        if recorded_epoch and current_epoch and recorded_epoch != current_epoch:
            return False
        if recorded_epoch != current_epoch:
            return None
        return start_time_fingerprints_match(
            recorded_start, current_start, tolerance=0) if exact else start_time_fingerprints_match(
                recorded_start, current_start)
    except (TypeError, ValueError, OSError, subprocess.SubprocessError):
        return None
