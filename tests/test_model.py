"""Model tests: the real 4B on this Mac. Skipped by default; run one at a time by hand:

    MACK_RUN_MODEL=1 .venv/bin/python -m pytest tests/test_model.py -k invoices -s

Each test appends its numbers to reports/model_results.json with a timestamp, so
FINDINGS.md can quote them. E2E-1 (kill/sleep/probe) and E2E-4 (pace table) are
scripts, because they need a separate worker process and minutes of probing:
scripts/e2e_tuesday.py and `mack bench`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest

from mack.apply import Refused, apply, plan, undo
from mack.batch import BatchRunner, BatchStore, create
from mack.config import ROOT, load_config
from mack.recipes import find
from mack.tasks import TaskStore

from .fixtures import make_fixtures as fx

pytestmark = pytest.mark.model
RESULTS = ROOT / "reports" / "model_results.json"


def record(name: str, data: dict) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    all_ = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    all_.append({"test": name, "at": time.strftime("%Y-%m-%d %H:%M"), **data})
    RESULTS.write_text(json.dumps(all_, indent=2))
    print(f"\n{name}: {json.dumps(data)}")


@pytest.fixture(scope="module")
def engine():
    from mack.engine import make_engine

    free = shutil.disk_usage(Path.home()).free / 1e9
    if free < 2.0:
        pytest.skip(f"only {free:.1f} GB free; model tests need 2 GB")
    e = make_engine(load_config(), load_draft=False)
    yield e
    del e
    gc.collect()  # unload before the next test file


@pytest.fixture
def env(tmp_path):
    cfg = load_config()
    cfg.paths.state = str(tmp_path / "state")
    cfg.paths.output = str(tmp_path / "output")
    src = load_config().state_dir / "calibration.json"
    if src.exists():
        shutil.copy(src, cfg.state_dir / "calibration.json")
    return cfg, TaskStore(cfg), BatchStore(cfg)


def run_job(engine, cfg, store, bstore, instruction, paths, **kw):
    task, _ = create(cfg, store, bstore, instruction, [str(p) for p in paths], **kw)
    runner = BatchRunner(engine, cfg, store, task, bstore)
    for wait in runner.steps():
        if wait > 0:
            time.sleep(wait)
    return store.load(task.id)


def from_recipe(engine, cfg, store, bstore, name, folder, pace="full", schema=None):
    r = find(name, cfg.recipes_dir)
    return run_job(engine, cfg, store, bstore, r.instruction, [folder], pace=pace,
                   schema=schema or r.resolved_schema(), recipe=r.name, glob=r.glob, ocr=r.ocr,
                   default_grounded=r.default_grounded, checks=r.checks, max_output_tokens=r.max_output_tokens)


INVOICE_SCHEMA = {"type": "object", "properties": {
    "vendor": {"type": "string", "x-grounded": True, "description": "the company that issued the invoice"},
    "date": {"type": "string", "format": "date", "x-grounded": True, "description": "the invoice date"},
    "amount": {"type": "number", "x-grounded": True, "description": "the total due"},
}}


def test_phase2_invoices_grounding(engine, env, tmp_path):
    cfg, store, bstore = env
    fx.invoices(tmp_path / "inv", 30)
    truth = json.loads((tmp_path / "inv-answers.json").read_text())
    task = run_job(engine, cfg, store, bstore, "Extract the vendor, invoice date and total due.", [tmp_path / "inv"],
                   pace="full", schema=INVOICE_SCHEMA)
    right = flagged = wrong_unflagged = failed = 0
    for r in bstore.items(task.id):
        t = truth[Path(r["path"]).name]
        if r["status"] == "failed":
            failed += 1
            continue
        out = json.loads(r["output"])
        ok = out["vendor"] == t["vendor"] and out["date"] == t["date"] and abs((out["amount"] or 0) - t["amount"]) < 0.01
        if r["status"] == "flagged":
            flagged += 1
        elif ok:
            right += 1
        else:
            wrong_unflagged += 1
    record("phase2_invoices", {"right": right, "flagged": flagged, "wrong_unflagged": wrong_unflagged, "failed": failed})
    assert wrong_unflagged <= 2


def test_phase5_recipes_on_mixed_fixtures(engine, env, tmp_path):
    cfg, store, bstore = env
    fx.mixed(tmp_path / "mix")
    results = {}
    tag = from_recipe(engine, cfg, store, bstore, "tag", tmp_path / "mix" / "notes")
    allowed = set(find("tag").options["tags"])
    rows = bstore.items(tag.id)
    results["tag"] = {"done": tag.batch["counts"]["done"], "flagged": tag.batch["counts"]["flagged"],
                      "bad_tags": sum(1 for r in rows if r["output"] and not set(json.loads(r["output"])["tags"]) <= allowed),
                      "note": "tags and summaries need a hand check in the review file"}
    s = from_recipe(engine, cfg, store, bstore, "summarize", tmp_path / "mix" / "notes")
    results["summarize"] = {k: s.batch["counts"][k] for k in ("done", "flagged", "failed")}
    c = from_recipe(engine, cfg, store, bstore, "clean", tmp_path / "mix" / "transcripts")
    results["clean"] = {k: c.batch["counts"][k] for k in ("done", "flagged", "failed")}
    rn = from_recipe(engine, cfg, store, bstore, "rename", tmp_path / "mix" / "docs", pace="full")
    truth = json.loads((tmp_path / "mix" / "docs-answers.json").read_text())
    right = wrong_unflagged = 0
    for r in bstore.items(rn.id):
        name = Path(r["path"]).name
        if name in truth and r["status"] == "done":
            out = json.loads(r["output"])
            if out["date"] == truth[name]["date"] and out["source"] and truth[name]["vendor"].split()[0].lower() in out["source"].lower():
                right += 1
            elif out["date"] not in (None, truth[name]["date"]):
                wrong_unflagged += 1
    results["rename"] = {"invoices_right": right, "wrong_date_unflagged": wrong_unflagged, **{k: rn.batch["counts"][k] for k in ("done", "flagged", "skipped")}}
    ex = from_recipe(engine, cfg, store, bstore, "extract", tmp_path / "mix" / "docs", schema=INVOICE_SCHEMA)
    results["extract"] = {k: ex.batch["counts"][k] for k in ("done", "flagged", "failed", "skipped")}
    record("phase5_recipes", results)
    assert results["rename"]["wrong_date_unflagged"] <= 2 and results["tag"]["bad_tags"] == 0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_e2e2_apply_undo_round_trip(engine, env, tmp_path):
    cfg, store, bstore = env
    fx.vault(tmp_path / "vault", 30)
    before = {p: sha(p) for p in (tmp_path / "vault").iterdir()}
    task = from_recipe(engine, cfg, store, bstore, "tag", tmp_path / "vault")
    changes, _ = apply(store, bstore, task)
    done_paths = {Path(r["path"]) for r in bstore.items(task.id) if r["status"] == "done"}
    changed = {p for p in before if sha(p) != before[p]}
    assert changed <= done_paths
    for p in changed:  # every change is inside the frontmatter block
        new, old = p.read_text(), p.read_bytes().decode()
        assert new.startswith("---\n") and new.split("\n---\n", 1)[1].endswith(old.split("\n---\n", 1)[-1][-200:])
    edited = sorted(changed)[0]
    edited.write_text(edited.read_text() + "\nhand edit\n")
    with pytest.raises(Refused):
        undo(store, task)
    undo(store, store.load(task.id), force=True)
    ok = all(sha(p) == before[p] for p in before if p != edited)
    record("e2e2_apply_undo", {"changed": len(changed), "restored_ok": ok})
    assert ok


def test_e2e3_rename_inbox(engine, env, tmp_path):
    cfg, store, bstore = env
    fx.inbox(tmp_path / "inbox")
    truth = json.loads((tmp_path / "inbox-answers.json").read_text())
    task = from_recipe(engine, cfg, store, bstore, "rename", tmp_path / "inbox", pace="steady")
    rows = {Path(r["path"]).name: r for r in bstore.items(task.id)}
    blanks_skipped = all(rows[f"blank{i}.png"]["reason"] == "no_text" for i in range(2))
    right = wrong_unflagged = 0
    for name, t in truth.items():
        r = rows[name]
        if r["status"] != "done":
            continue
        out = json.loads(r["output"])
        if out["date"] == t["date"]:
            right += out["source"] is not None and t["vendor"].split()[0].lower() in (out["source"] or "").lower()
        elif out["date"] is not None:
            wrong_unflagged += 1
    changes, _ = plan(store, bstore, task)
    record("e2e3_rename_inbox", {"blanks_skipped": blanks_skipped, "invoices_right": right, "wrong_date_unflagged": wrong_unflagged,
                                 "renames_planned": len(changes), **task.batch["counts"]})
    assert blanks_skipped and wrong_unflagged == 0


def test_e2e5_never_answers_from_memory(engine, env, tmp_path):
    cfg, store, bstore = env
    fx.entities(tmp_path / "ent")
    schema = {"type": "object", "properties": {
        "capital": {"type": "string", "x-grounded": True, "description": "the capital city of the place named in the text"},
        "founder": {"type": "string", "x-grounded": True, "description": "who founded the company named in the text"},
        "birth_year": {"type": "integer", "x-grounded": True, "description": "the birth year of the person named in the text"},
    }}
    task = run_job(engine, cfg, store, bstore, "Extract the requested facts about the entity mentioned.", [tmp_path / "ent"],
                   pace="full", schema=schema)
    leaked = model_null = caught = 0
    for r in bstore.items(task.id):
        out = json.loads(r["output"]) if r["output"] else {}
        if any(out.get(k) not in (None, "") for k in schema["properties"]):
            leaked += 1
        elif r["status"] == "flagged":
            caught += 1
        else:
            model_null += 1
    record("e2e5_no_memory", {"returned_null_itself": model_null, "caught_by_grounding": caught, "leaked": leaked})
    assert leaked == 0
