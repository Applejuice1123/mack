"""Phase 3: the CLI, run as real subprocesses against the stub engine."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path):
    (tmp_path / "mack.toml").write_text(
        f'[paths]\nstate = "{tmp_path}/state"\noutput = "{tmp_path}/output"\nrecipes = "{tmp_path}/recipes"\n'
        "[batch]\nitem_rest_s = 0.0\n"
    )
    e = dict(os.environ)
    e.update(MACK_CONFIG=str(tmp_path / "mack.toml"), MACK_STUB_ENGINE="1", PYTHONPATH=str(ROOT), MACK_FAKE_POWER="ac")
    e.pop("MACK_STUB_RESPONDER", None)
    return tmp_path, e


def mack(env, *args, responder=None, extra_env=None):
    tmp, e = env
    e = dict(e)
    if responder:
        e["MACK_STUB_RESPONDER"] = f"tests.stub_responders:{responder}"
    e.update(extra_env or {})
    return subprocess.run([sys.executable, "-m", "mack.cli", *args], capture_output=True, text=True, env=e, cwd=str(ROOT), timeout=120)


def notes(folder: Path, n: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = folder / f"note{i:03d}.md"
        p.write_text(f"Note {i} is about the weekly meeting and topic {i}.")
        out.append(p)
    return out


def invoices(folder: Path) -> dict[Path, str]:
    folder.mkdir(parents=True, exist_ok=True)
    data = {"a.txt": ("Northwind", "2026-03-03"), "b.txt": ("Contoso", "2026-04-11"), "c.txt": ("Northwind", "2026-03-03")}
    files = {}
    for name, (vendor, date) in data.items():
        p = folder / name
        p.write_text(f"INVOICE\nVendor: {vendor}\nDate: {date}\nTotal: $120.00\n")
        files[p] = sha(p)
    return files


def job_id(stdout: str) -> str:
    return next(line.split()[1] for line in stdout.splitlines() if line.startswith("job "))


def test_dry_run_plans_without_creating_a_job(env):
    tmp, _ = env
    notes(tmp / "vault", 12)
    r = mack(env, "run", "--dry-run", "Summarize.", str(tmp / "vault"))
    assert r.returncode == 0, r.stderr
    assert "12 items" in r.stdout and "note000.md" in r.stdout
    assert not (tmp / "state" / "tasks").exists() or not any((tmp / "state" / "tasks").iterdir())


def test_limit_wait_status(env):
    tmp, _ = env
    notes(tmp / "vault", 9)
    r = mack(env, "run", "--limit", "5", "--wait", "--pace", "full", "Echo it.", str(tmp / "vault"))
    assert r.returncode == 0, r.stderr + r.stdout
    jid = job_id(r.stdout)
    out = tmp / "output" / f"{jid}.jsonl"
    assert len(out.read_text().splitlines()) == 5
    s = mack(env, "status")
    assert jid in s.stdout and "5/5" in s.stdout
    d = mack(env, "status", jid)
    assert "5/5" in d.stdout and "5 done" in d.stdout


def test_review_shows_flagged_section(env, tmp_path):
    tmp, _ = env
    notes(tmp / "vault", 4)
    schema = tmp / "schema.json"
    schema.write_text(json.dumps({"type": "object", "properties": {"text": {"type": "string", "x-grounded": True}}}))
    r = mack(env, "run", "--wait", "--pace", "full", "--schema", str(schema), "First word.", str(tmp / "vault"), responder="flag_note_002")
    jid = job_id(r.stdout)
    rv = mack(env, "review", jid, "--no-open")
    assert rv.returncode == 0
    review = Path(rv.stdout.strip()).read_text()
    head = review.split("## Done")[0]
    assert "## Needs attention" in head and "note002.md" in head and "Flags: text" in head


def test_rename_apply_and_undo_round_trip(env):
    tmp, _ = env
    files = invoices(tmp / "inbox")
    r = mack(env, "run", "--wait", "--pace", "full", "--recipe", "rename", str(tmp / "inbox"), responder="invoice_fields")
    assert r.returncode == 0, r.stderr + r.stdout
    jid = job_id(r.stdout)

    dry = mack(env, "apply", jid, "--dry-run")
    assert "a.txt → 2026-03-03 Northwind Monthly invoice.txt" in dry.stdout
    assert "c.txt → 2026-03-03 Northwind Monthly invoice-2.txt" in dry.stdout  # collision
    assert sorted(p.name for p in (tmp / "inbox").iterdir()) == ["a.txt", "b.txt", "c.txt"]

    ap = mack(env, "apply", jid)
    assert ap.returncode == 0, ap.stderr
    names = sorted(p.name for p in (tmp / "inbox").iterdir())
    assert "2026-04-11 Contoso Monthly invoice.txt" in names and "a.txt" not in names

    un = mack(env, "undo", jid)
    assert un.returncode == 0, un.stderr
    assert {p: sha(p) for p in files} == files


def test_apply_refuses_when_a_file_changed_after_the_run(env):
    tmp, _ = env
    files = invoices(tmp / "inbox")
    jid = job_id(mack(env, "run", "--wait", "--pace", "full", "--recipe", "rename", str(tmp / "inbox"), responder="invoice_fields").stdout)
    victim = tmp / "inbox" / "b.txt"
    victim.write_text("edited by hand")
    ap = mack(env, "apply", jid)
    assert ap.returncode == 3 and "b.txt" in ap.stderr
    assert sorted(p.name for p in (tmp / "inbox").iterdir()) == ["a.txt", "b.txt", "c.txt"]
    assert sha(tmp / "inbox" / "a.txt") == files[tmp / "inbox" / "a.txt"]


def test_undo_refuses_after_post_apply_edit_and_force_restores_the_rest(env):
    tmp, _ = env
    vault = tmp / "vault"
    paths = notes(vault, 3)
    before = {p: sha(p) for p in paths}
    jid = job_id(mack(env, "run", "--wait", "--pace", "full", "--recipe", "tag", str(vault), responder="tagger").stdout)
    assert mack(env, "apply", jid).returncode == 0
    assert paths[0].read_text().startswith("---\ntags: [meeting]")
    paths[1].write_text(paths[1].read_text() + "\nmy own edit")
    edited = sha(paths[1])

    un = mack(env, "undo", jid)
    assert un.returncode == 3 and "note001.md" in un.stderr
    assert not paths[0].read_text().startswith("Note 0")  # nothing was undone

    forced = mack(env, "undo", jid, "--force")
    assert forced.returncode == 0
    assert sha(paths[0]) == before[paths[0]] and sha(paths[2]) == before[paths[2]]
    assert sha(paths[1]) == edited  # the hand-edited file is left alone


def test_undo_log_is_on_disk_before_any_file_is_touched(env):
    tmp, _ = env
    vault = tmp / "vault"
    paths = notes(vault, 3)
    before = {p: sha(p) for p in paths}
    jid = job_id(mack(env, "run", "--wait", "--pace", "full", "--recipe", "tag", str(vault), responder="tagger").stdout)
    crash = mack(env, "apply", jid, extra_env={"MACK_TEST_CRASH_AFTER_UNDO_LOG": "1"})
    assert crash.returncode == 9  # killed between writing the log and the first change
    log = tmp / "output" / f"{jid}.undo.jsonl"
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(entries) == 3 and all(e["op"] == "modify" and e["original_b64"] for e in entries)
    assert {p: sha(p) for p in paths} == before


def test_errors_have_exit_codes(env):
    tmp, _ = env
    assert mack(env, "run", "only-an-instruction").returncode == 1
    assert mack(env, "run", "--recipe", "nope", str(tmp)).returncode == 1
    bad = tmp / "bad.json"
    bad.write_text('{"type": "object", "properties": {"x": {"type": "object"}}}')
    r = mack(env, "run", "--schema", str(bad), "x", str(tmp))
    assert r.returncode == 1 and "not supported" in r.stderr
