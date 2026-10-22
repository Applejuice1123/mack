"""Phase 1: the batch job core. Guarantee 3 (durable): kill, resume, nothing lost or repeated."""

import json

from mack.batch import output_path, write_review

from .batch_helpers import Killed, create, first_word, jsonl, make_env, quiet_mac, run, write_notes  # noqa: F401


def test_fifty_files_all_run(tmp_path):
    cfg, store, bstore, engine = make_env(tmp_path)
    write_notes(tmp_path / "vault", 50)
    task, plan = create(cfg, store, bstore, "Echo the first words.", [str(tmp_path / "vault")], pace="full")
    assert plan["items"] == 50 and "50 items" in plan["text"]
    run(cfg, store, bstore, engine, task)
    t = store.load(task.id)
    assert t.status == "done" and t.batch["counts"]["done"] == 50
    lines = jsonl(output_path(store, t))
    assert len(lines) == 50 and all(l["status"] == "done" for l in lines)
    assert t.summary()["units_done"] == 50 and t.progress == 1.0


def test_kill_after_item_23_then_resume(tmp_path):
    calls = {"n": 0}

    def dies_once(prompt):
        calls["n"] += 1
        if calls["n"] == 24:  # the 24th generation is item index 23
            raise Killed()
        return first_word(prompt)

    cfg, store, bstore, engine = make_env(tmp_path, responder=dies_once)
    write_notes(tmp_path / "vault", 50)
    task, _ = create(cfg, store, bstore, "First word.", [str(tmp_path / "vault")], pace="full")
    try:
        run(cfg, store, bstore, engine, task)
        raise AssertionError("the stub should have killed the run")
    except Killed:
        pass
    assert bstore.item(task.id, 23)["status"] == "running"  # exactly what a dead process leaves behind
    assert bstore.counts(task.id)["done"] == 23

    run(cfg, store, bstore, engine, task)  # resume
    t = store.load(task.id)
    assert t.status == "done"
    lines = jsonl(output_path(store, t))
    assert len(lines) == 50 and len({l["path"] for l in lines}) == 50
    assert bstore.item(task.id, 23)["attempts"] == 2
    assert bstore.item(task.id, 24)["attempts"] == 1


def test_item_that_keeps_killing_the_worker_fails_after_two_attempts(tmp_path):
    def always_dies_on_note_007(prompt):
        if "Note 7 " in prompt:
            raise Killed()
        return first_word(prompt)

    cfg, store, bstore, engine = make_env(tmp_path, responder=always_dies_on_note_007)
    write_notes(tmp_path / "vault", 10)
    task, _ = create(cfg, store, bstore, "First word.", [str(tmp_path / "vault")], pace="full")
    for _ in range(2):
        try:
            run(cfg, store, bstore, engine, task)
        except Killed:
            pass
    run(cfg, store, bstore, engine, task)
    row = bstore.item(task.id, 7)
    assert row["status"] == "failed" and "2 attempts" in row["error"]
    assert store.load(task.id).status == "done"


def test_changed_file_is_skipped_not_processed_stale(tmp_path):
    cfg, store, bstore, engine = make_env(tmp_path)
    paths = write_notes(tmp_path / "vault", 5)
    task, _ = create(cfg, store, bstore, "First word.", [str(tmp_path / "vault")], pace="full")
    paths[2].write_text("Edited after enqueue.")
    run(cfg, store, bstore, engine, task)
    row = bstore.item(task.id, 2)
    assert row["status"] == "skipped" and row["reason"] == "changed_since_enqueue"
    assert len(engine.calls) == 4


def test_oversized_and_binary_files_are_skipped_with_reasons(tmp_path):
    cfg, store, bstore, engine = make_env(tmp_path)
    vault = tmp_path / "vault"
    write_notes(vault, 3)
    (vault / "huge.txt").write_text("word " * 1_000_000)  # 5 MB
    (vault / "picture.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 40)
    task, plan = create(cfg, store, bstore, "First word.", [str(vault / "huge.txt"), str(vault / "picture.png"), str(vault)], pace="full")
    assert plan["skipped"] == 2
    run(cfg, store, bstore, engine, task)
    rows = {r["path"].rsplit("/", 1)[1]: r for r in bstore.items(task.id)}
    assert rows["huge.txt"]["status"] == "skipped" and rows["huge.txt"]["reason"].startswith("too_large")
    assert rows["picture.png"]["status"] == "skipped" and rows["picture.png"]["reason"].startswith("unsupported_type")
    assert store.load(task.id).status == "done" and bstore.counts(task.id)["done"] == 3


def test_three_engine_failures_in_a_row_pause_the_job(tmp_path):
    def broken(prompt):
        raise RuntimeError("Metal command buffer failed")

    cfg, store, bstore, engine = make_env(tmp_path, responder=broken)
    write_notes(tmp_path / "vault", 8)
    task, _ = create(cfg, store, bstore, "First word.", [str(tmp_path / "vault")], pace="full")
    run(cfg, store, bstore, engine, task)
    t = store.load(task.id)
    c = bstore.counts(task.id)
    assert t.status == "paused" and "3 engine failures" in t.error
    assert c["failed"] == 3 and c["pending"] == 5
    assert store.next_runnable(cfg) is None  # stays paused until the user resumes it


def test_identical_files_get_independent_outputs(tmp_path):
    cfg, store, bstore, engine = make_env(tmp_path, responder=first_word)
    vault = tmp_path / "vault"
    vault.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (vault / name).write_text("Same words in every file.")
    (vault / "d.md").write_text("Different words here.")
    task, _ = create(cfg, store, bstore, "Echo the first word.", [str(vault)], pace="full")
    run(cfg, store, bstore, engine, task)
    outs = [json.loads(r["output"])["text"] for r in bstore.items(task.id)]
    assert outs == ["Same", "Same", "Same", "Different"]
    # each item's prompt held exactly one file
    assert all(c["prompt"].count("<<<FILE\n") == 1 for c in engine.calls)


def test_review_file_lists_attention_items_first(tmp_path):
    cfg, store, bstore, engine = make_env(tmp_path)
    write_notes(tmp_path / "vault", 3)
    task, _ = create(cfg, store, bstore, "First word.", [str(tmp_path / "vault")], pace="full")
    run(cfg, store, bstore, engine, task)
    bstore.update(task.id, 1, status="flagged", flags='["text"]')
    text = write_review(store, bstore, store.load(task.id)).read_text()
    assert text.index("## Needs attention") < text.index("## Done")
    assert "note001.md" in text.split("## Done")[0]
