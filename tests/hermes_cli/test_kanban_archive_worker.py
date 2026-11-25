"""Archive worker termination and audit contracts (regression for #76196)."""

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_archive_running_task_terminates_worker(kanban_home, monkeypatch):
    """``archive_task`` on a *running* task must actually signal its host-local
    worker process, not just null ``worker_pid`` in the DB (#76196: a worker
    kept running past its own archive and could still push/complete work
    against a task nothing tracks anymore). The termination outcome is
    auditable via the ``archive_worker_termination`` event."""
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        # A verified spawn: an uncaptured fingerprint would (correctly) refuse the signal.
        monkeypatch.setattr(kbd, "_process_fingerprint", lambda _pid: "boot:1|777")
        kbd._set_worker_pid(conn, t, 54321)

        alive = [True]
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: alive[0])
        signalled = []

        def terminate(pid, sig):
            signalled.append((pid, sig))
            alive[0] = False

        assert kb.archive_task(conn, t, signal_fn=terminate) is True

        assert signalled and signalled[0][0] == 54321

        row = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'archive_worker_termination'",
            (t,),
        ).fetchone()
        payload = json.loads(row["payload"])
        assert payload["prev_pid"] == 54321
        assert payload["host_local"] is True
        assert payload["termination_attempted"] is True
        assert payload["terminated"] is True
        assert kb.get_task(conn, t).status == "archived"


def test_archive_non_running_task_does_not_attempt_termination(kanban_home):
    """A never-claimed (``triage``/``ready``/``done``) task has no live worker:
    ``archive_task`` must not signal anything, and no termination event is
    recorded — only for tasks that were actually ``running`` at archive time."""
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        signalled = []
        assert kb.archive_task(
            conn, t, signal_fn=lambda pid, sig: signalled.append((pid, sig)),
        ) is True
        assert signalled == []
        row = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND kind = 'archive_worker_termination'",
            (t,),
        ).fetchone()
        assert row is None
