import json
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli.kanban_quota_cooldown import MARKER, emit_reset, read_reset


def test_marker_roundtrip_is_bound_to_worker_pid(capsys, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    emit_reset({"failure_resets_at": 5000, "error": "private error never emitted"})
    log = capsys.readouterr().err
    assert "private" not in log
    assert read_reset(log, os.getpid(), 7) == {"quota_exhausted": True, "quota_resets_at": 5000}
    assert read_reset(log, os.getpid() + 1, 7)["quota_resets_at"] is None
    assert read_reset(log, os.getpid(), 8)["quota_resets_at"] is None


@pytest.mark.parametrize("reset", [None, "tomorrow", True, float("nan"), -1, 10**500])
def test_invalid_reset_is_explicit_unknown(reset):
    raw = MARKER + json.dumps({"pid": 123, "run_id": "7", "quota_exhausted": True, "resets_at": reset})
    assert read_reset(raw, 123, 7) == {"quota_exhausted": True, "quota_resets_at": None}


@pytest.mark.parametrize("reset", [5000, None])
def test_durable_guard_survives_restart_and_never_spawns_before_reset(tmp_path, monkeypatch, reset, all_assignees_spawnable):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    path = tmp_path / "kanban.db"
    with kbc.connect(path) as conn:
        tid = kb.create_task(conn, title="quota", assignee="worker")
        kb.claim_task(conn, tid)
        rid = kb.get_task(conn, tid).current_run_id
        conn.execute("UPDATE task_runs SET outcome='rate_limited',status='rate_limited',ended_at=1000,metadata=? WHERE id=?",
                     (json.dumps({"quota_exhausted": True, "quota_resets_at": reset}), rid))
        conn.execute("UPDATE tasks SET status='ready',current_run_id=NULL,claim_lock=NULL,claim_expires=NULL,worker_pid=NULL WHERE id=?", (tid,))
        conn.commit()
    launches = []
    for now in (1200, 2000, 4999):
        monkeypatch.setattr(dispatch.time, "time", lambda: now)
        with kbc.connect(path) as conn:
            result = dispatch.dispatch_once(conn, spawn_fn=lambda *a, **kw: launches.append(a) or 12345)
            assert not result.spawned
            assert dispatch.check_respawn_guard(conn, tid) is not None
    assert launches == []
    monkeypatch.setattr(dispatch.time, "time", lambda: 5000)
    with kbc.connect(path) as conn:
        assert dispatch.check_respawn_guard(conn, tid) == (None if reset else "quota_reset_unknown")
        # Exercise the actual dispatch gate, not just its helper predicate.
        if reset:
            monkeypatch.setattr(dispatch, "_set_worker_pid", lambda *a, **kw: True)
            result = dispatch.dispatch_once(conn, spawn_fn=lambda *a, **kw: launches.append(a) or 12345)
            assert len(result.spawned) == 1, vars(result)
            assert len(launches) == 1
            assert kb.claim_task(conn, tid, claimer="second") is None


def test_classifier_persists_quota_metadata(monkeypatch):
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda pid: ("rate_limited", 75))
    monkeypatch.setattr(kb, "read_worker_log", lambda *a, **kw:
                        MARKER + json.dumps({"pid": 123, "run_id": "7", "quota_exhausted": True, "resets_at": 5000}))
    result = dispatch._classify_dead_worker(123, "host:owner", task_id="task", run_id=7)
    assert result.event_payload["quota_resets_at"] == 5000
    assert result.rate_limited


@pytest.mark.parametrize("log", ["", MARKER + "{broken", MARKER + "null"])
def test_missing_or_corrupt_marker_is_conservative(log):
    assert read_reset(log, 123, 7) == {"quota_exhausted": True, "quota_resets_at": None}


def test_transient_error_keeps_existing_cooldown_contract(capsys, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    emit_reset({"failure_reason": "server_error"})
    assert read_reset(capsys.readouterr().err, os.getpid(), 7)["quota_exhausted"] is False


def test_unreadable_worker_log_is_conservative(monkeypatch):
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda pid: ("rate_limited", 75))
    def unreadable(*a, **kw):
        raise OSError("unavailable")
    monkeypatch.setattr(kb, "read_worker_log", unreadable)
    result = dispatch._classify_dead_worker(123, "host:owner", task_id="task", run_id=7)
    assert result.event_payload["quota_exhausted"] is True
    assert result.event_payload["quota_resets_at"] is None


def test_emitted_reset_is_persisted_by_reclaim_and_reopened_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(dispatch.time, "time", lambda: 1000)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda pid: ("rate_limited", 75))
    path = tmp_path / "kanban.db"
    with kbc.connect(path) as conn:
        tid = kb.create_task(conn, title="integration", assignee="worker")
        kb.claim_task(conn, tid)
        run = kb.get_task(conn, tid).current_run_id
        conn.execute("UPDATE tasks SET worker_pid=123 WHERE id=?", (tid,))
        conn.commit()
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
        monkeypatch.setattr(os, "getpid", lambda: 123)
        emit_reset({"failure_reason": "rate_limit", "failure_resets_at": 5000})
        log = capsys.readouterr().err
        monkeypatch.setattr(kb, "read_worker_log", lambda *a, **kw: log)
        assert tid in dispatch._reclaim_dead_workers(conn).rate_limited
        metadata = json.loads(conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run,)).fetchone()[0])
        assert metadata["quota_resets_at"] == 5000
    monkeypatch.setattr(dispatch.time, "time", lambda: 4000)
    with kbc.connect(path) as reopened:
        assert dispatch.check_respawn_guard(reopened, tid) == "rate_limit_cooldown"
