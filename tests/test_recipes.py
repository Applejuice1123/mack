"""Phase 5: recipes, apply rules and OCR inputs."""

import json
from pathlib import Path

import pytest

from mack import textsource
from mack.apply import apply, merge_frontmatter, plan, undo
from mack.recipes import RecipeError, all_recipes, find, load

from .batch_helpers import create, file_text, make_env, quiet_mac, run  # noqa: F401
from . import stub_responders


def job_from_recipe(tmp_path, name, files: dict, responder, **kw):
    cfg, store, bstore, engine = make_env(tmp_path, responder=responder)
    folder = tmp_path / "in"
    folder.mkdir(exist_ok=True)
    for fname, content in files.items():
        p = folder / fname
        p.write_bytes(content) if isinstance(content, bytes) else p.write_text(content)
    r = find(name, tmp_path / "recipes")
    task, plan_info = create(
        cfg, store, bstore, r.instruction, [str(folder)], pace="full",
        schema=kw.pop("schema", None) or r.resolved_schema(kw.pop("options", None)), recipe=r.name, glob=r.glob,
        ocr=r.ocr, default_grounded=r.default_grounded, checks=r.checks, max_output_tokens=r.max_output_tokens,
    )
    run(cfg, store, bstore, engine, task)
    return cfg, store, bstore, store.load(task.id), folder


def test_every_builtin_recipe_loads_and_validates():
    names = {r.name for r in all_recipes(Path("/nonexistent"))}
    assert names == {"tag", "summarize", "rename", "extract", "clean"}
    with pytest.raises(RecipeError):
        find("extract", Path("/nonexistent")).resolved_schema()  # needs --schema


@pytest.mark.parametrize("name,files,reply,expect", [
    ("tag", {"a.md": "Weekly meeting notes.", "b.md": "An idea for a game.", "c.md": "Meeting with Sam."},
     stub_responders.tagger, "frontmatter + tags, summary"),
    ("summarize", {"a.txt": "The bridge opened in 1932. It carries four lanes.", "b.txt": "Rain fell all week.", "c.txt": "Cats sleep a lot."},
     lambda p: json.dumps({"summary": "Short.", "key_points": ["first"], "key_points_quotes": [file_text(p).split(".")[0]]}), "new file"),
    ("rename", {"a.txt": "Vendor: Northwind\nDate: 2026-03-03\n", "b.txt": "Vendor: Contoso\nDate: 2026-01-09\n", "c.txt": "Vendor: Fabrikam\nDate: 2026-02-02\n"},
     stub_responders.invoice_fields, "→ 2026-"),
    ("clean", {"a.txt": "um so the the plan is uh ready", "b.txt": "we we need tests", "c.txt": "like ship it friday"},
     lambda p: json.dumps({"cleaned": file_text(p).replace("um ", "").replace("uh ", ""), "action_items": []}), "new file"),
])
def test_recipe_runs_on_three_files_and_plans_changes(tmp_path, name, files, reply, expect):
    cfg, store, bstore, task, folder = job_from_recipe(tmp_path, name, files, reply)
    assert task.batch["counts"]["done"] == 3, [dict(r) for r in bstore.items(task.id)]
    changes, _ = plan(store, bstore, task)
    assert len(changes) == 3 and all(expect in c.summary for c in changes)


def test_extract_with_user_schema_and_csv(tmp_path):
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "year": {"type": "integer"}}}
    files = {"a.txt": "Founded in Lisbon in 1911.", "b.txt": "Moved to Porto in 1950.", "c.txt": "Nothing here."}

    def reply(p):
        t = file_text(p)
        import re
        city = re.search(r"in ([A-Z]\w+)", t)
        year = re.search(r"\d{4}", t)
        return json.dumps({"city": city.group(1) if city else None, "year": int(year.group(0)) if year else None})

    cfg, store, bstore, task, folder = job_from_recipe(tmp_path, "extract", files, reply, schema=schema)
    assert task.batch["default_grounded"] is True
    out_csv = tmp_path / "places.csv"
    changes, _ = apply(store, bstore, task, csv_path=str(out_csv))
    assert out_csv.read_text().splitlines()[0] == "path,city,year"
    assert "Lisbon,1911" in out_csv.read_text()
    undo(store, task)
    assert not out_csv.exists()


def test_tag_frontmatter_merge_and_exact_undo(tmp_path):
    with_front = "---\ntitle: Standup\ntags: [old]\naliases:\n  - daily\n---\nWeekly meeting body.\n"
    files = {"a.md": with_front, "b.md": "An idea without frontmatter.\n"}
    cfg, store, bstore, task, folder = job_from_recipe(tmp_path, "tag", files, stub_responders.tagger)
    before = {p: p.read_bytes() for p in folder.iterdir()}
    apply(store, bstore, task)
    a = (folder / "a.md").read_text()
    assert a.startswith("---\ntitle: Standup\ntags: [old]\naliases:\n  - daily\nsummary: ")  # existing keys kept, byte for byte
    assert a.endswith("---\nWeekly meeting body.\n") and "tags: [meeting]" not in a
    b = (folder / "b.md").read_text()
    assert b.startswith("---\ntags: [idea]\nsummary: An idea without frontmatter\n---\nAn idea without frontmatter.")
    undo(store, task)
    assert {p: p.read_bytes() for p in folder.iterdir()} == before


def test_frontmatter_overwrite_replaces_only_that_key():
    text = "---\ntitle: X\ntags:\n  - a\n  - b\nkeep: 1\n---\nbody\n"
    new, written = merge_frontmatter(text, {"tags": ["c"]}, overwrite=True)
    assert new == "---\ntitle: X\nkeep: 1\ntags: [c]\n---\nbody\n" and written == ["tags"]


def test_rename_collision_and_undated(tmp_path):
    files = {"x.txt": "Vendor: Northwind\nDate: 2026-03-03\n", "y.txt": "Vendor: Northwind\nDate: 2026-03-03\n", "z.txt": "Vendor: Contoso\n"}
    cfg, store, bstore, task, folder = job_from_recipe(tmp_path, "rename", files, stub_responders.invoice_fields)
    names = sorted(c.new_path.name for c in plan(store, bstore, task)[0])
    assert names == ["2026-03-03 Northwind Monthly invoice-2.txt", "2026-03-03 Northwind Monthly invoice.txt", "undated Contoso Monthly invoice.txt"]


def test_rename_refuses_a_date_the_file_does_not_contain(tmp_path):
    files = {"x.txt": "Vendor: Northwind\nDate: 2026-03-03\n"}
    wrong = lambda p: json.dumps({"date": "2025-12-25", "source": "Northwind", "kind": "invoice", "title": "Bill"})
    cfg, store, bstore, task, folder = job_from_recipe(tmp_path, "rename", files, wrong)
    row = bstore.items(task.id)[0]
    assert row["status"] == "flagged" and json.loads(row["output"])["date"] is None
    assert plan(store, bstore, task)[0] == []  # flagged items are not applied by default


def test_clean_that_drops_words_is_flagged_not_applied(tmp_path):
    original = "so the quarterly numbers came in higher than expected because the new pricing landed in march and churn fell"
    gutted = lambda p: json.dumps({"cleaned": "The quarterly numbers came in higher.", "action_items": []})
    cfg, store, bstore, task, folder = job_from_recipe(tmp_path, "clean", {"t.txt": original}, gutted)
    row = bstore.items(task.id)[0]
    assert row["status"] == "flagged" and "cleaned" in json.loads(row["flags"])
    assert json.loads(row["output"])["cleaned_retention"] < 0.85
    assert plan(store, bstore, task)[0] == []


def test_summarize_unquoted_key_point_is_dropped_and_flagged(tmp_path):
    reply = lambda p: json.dumps({"summary": "A bridge.", "key_points": ["opened 1932", "made of gold"],
                                  "key_points_quotes": ["The bridge opened in 1932", "It is made of gold"]})
    cfg, store, bstore, task, folder = job_from_recipe(tmp_path, "summarize", {"a.txt": "The bridge opened in 1932. It carries four lanes."}, reply)
    row = bstore.items(task.id)[0]
    out = json.loads(row["output"])
    assert out["key_points"] == ["opened 1932"] and row["status"] == "flagged"


def test_user_recipe_shadows_builtin(tmp_path):
    (tmp_path / "recipes").mkdir()
    builtin = find("tag", tmp_path / "recipes")
    custom = builtin.path.read_text().replace("[meeting, idea, bug, reference, personal]", "[work, home]")
    (tmp_path / "recipes" / "tag.yaml").write_text(custom)
    r = find("tag", tmp_path / "recipes")
    assert not r.builtin and r.resolved_schema()["properties"]["tags"]["items"]["enum"] == ["work", "home"]
    assert r.resolved_schema({"tags": ["x"]})["properties"]["tags"]["items"]["enum"] == ["x"]
    assert {x.name: x.builtin for x in all_recipes(tmp_path / "recipes")}["tag"] is False


def test_ocr_is_used_for_images_and_blank_images_are_skipped(tmp_path, monkeypatch):
    seen = []

    def fake_ocr(path):
        seen.append(Path(path).name)
        return "INVOICE\nVendor: Northwind\nDate: 2026-03-03\nTotal due $120.00" if "bill" in Path(path).name else "  "

    monkeypatch.setattr(textsource, "ocr_image", fake_ocr)
    files = {"bill.png": b"\x89PNG fake", "blank.png": b"\x89PNG blank"}
    cfg, store, bstore, task, folder = job_from_recipe(tmp_path, "rename", files, stub_responders.invoice_fields)
    rows = {Path(r["path"]).name: r for r in bstore.items(task.id)}
    assert sorted(seen) == ["bill.png", "blank.png"]
    assert rows["bill.png"]["status"] == "done" and rows["blank.png"]["reason"] == "no_text"
    assert rows["bill.png"]["text"].startswith("INVOICE")  # cached at enqueue, not recognized twice
    assert len(seen) == 2


def test_bad_recipe_files_are_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\ninstruction: x\napply: {kind: email}\n")
    with pytest.raises(RecipeError):
        load(p)
    p.write_text("name: bad\ninstruction: x\napply: {kind: none}\nsurprise: 1\n")
    with pytest.raises(RecipeError):
        load(p)
