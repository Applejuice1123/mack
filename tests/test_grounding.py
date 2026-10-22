"""Phase 2: schema-constrained outputs and grounding checks. Guarantee 1 (grounded)."""

import json

import pytest

from mack.schema import PREAMBLE, SchemaError, appears, dates_in, extract_json, ground, norm, parse_schema, validate

from .batch_helpers import create, file_text, make_env, quiet_mac, run  # noqa: F401

INVOICE = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string", "x-grounded": True},
        "date": {"type": "string", "format": "date", "x-grounded": True},
        "amount": {"type": "number", "x-grounded": True},
        "kind": {"enum": ["invoice", "receipt", "other"]},
        "note": {"type": "string", "x-quote": True},
    },
}
TEXT = "INVOICE from Northwind Traders\nDate: March 3, 2026\nTotal due: $1,240.50\nThanks for your business, see you in spring."


def job(tmp_path, reply, schema=INVOICE, text=TEXT):
    replies = reply if isinstance(reply, list) else [reply]
    it = iter(replies)
    cfg, store, bstore, engine = make_env(tmp_path, responder=lambda p: next(it) if len(replies) > 1 else replies[0])
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "inv.txt").write_text(text)
    task, _ = create(cfg, store, bstore, "Extract the invoice fields.", [str(tmp_path / "in")], pace="full", schema=schema)
    run(cfg, store, bstore, engine, task)
    return bstore.items(task.id)[0], engine


def good(**over):
    obj = {"vendor": "Northwind Traders", "date": "2026-03-03", "amount": 1240.5, "kind": "invoice",
           "note": "thanks", "note_quote": "Thanks for your business"}
    obj.update(over)
    return json.dumps(obj)


def test_valid_output_is_done(tmp_path):
    row, _ = job(tmp_path, good())
    assert row["status"] == "done", row["reason"]
    assert json.loads(row["output"])["vendor"] == "Northwind Traders"


def test_prose_instead_of_json_fails_after_one_retry(tmp_path):
    row, engine = job(tmp_path, "The vendor is Northwind Traders.")
    assert row["status"] == "failed" and row["reason"].startswith("invalid_json")
    assert len(engine.calls) == 2 and "previous reply was rejected" in engine.calls[1]["prompt"]


def test_enum_outside_the_list_fails_schema_after_retry(tmp_path):
    row, engine = job(tmp_path, good(kind="purchase order"))
    assert row["status"] == "failed" and row["reason"].startswith("schema")
    assert len(engine.calls) == 2


def test_retry_can_recover(tmp_path):
    row, _ = job(tmp_path, ["not json", good()])
    assert row["status"] == "done"


def test_invented_vendor_is_nulled_and_flagged(tmp_path):
    row, _ = job(tmp_path, good(vendor="Acme"))
    out = json.loads(row["output"])
    assert out["vendor"] is None and json.loads(row["flags"]) == ["vendor"] and row["status"] == "flagged"


def test_fabricated_quote_nulls_value_and_quote(tmp_path):
    row, _ = job(tmp_path, good(note="they thanked us", note_quote="We appreciate your loyalty"))
    out = json.loads(row["output"])
    assert out["note"] is None and out["note_quote"] is None and "note" in json.loads(row["flags"])


def test_grounding_ignores_case_and_whitespace():
    assert appears("ACME  Corp", norm("Payment to acme corp\nreceived"))
    assert not appears("Acme Inc", norm("Payment to acme corp"))


def test_dates_numbers_and_wrong_values():
    schema = parse_schema(INVOICE)
    out = validate(schema, json.loads(good(date="2026-03-04", amount=1240.5)))
    flags = ground(schema, out, TEXT)
    assert flags == ["date"] and out["amount"] == 1240.5  # "$1,240.50" in the text counts as 1240.5
    found = dates_in("Due 03/04/2026 or 4 March 2026")
    from datetime import date
    assert found[date(2026, 3, 4)] is True  # "4 March 2026" settles it
    assert found[date(2026, 4, 3)] is False  # only the ambiguous numeric form supports 3 April


def test_ambiguous_numeric_date_is_flagged_not_trusted():
    schema = parse_schema(INVOICE)
    text = "INVOICE from Northwind Traders\nInvoice date: 03/09/2025\nTotal due: $1,240.50"
    out = validate(schema, json.loads(good(date="2025-03-09", note=None, note_quote=None)))
    flags = ground(schema, out, text)
    assert flags == ["date"] and out["date"] == "2025-03-09" and out["_ambiguous"] == ["date"]
    unambiguous = "Invoice date: 25/09/2025 from Northwind Traders, total $1,240.50"
    out2 = validate(schema, json.loads(good(date="2025-09-25", note=None, note_quote=None)))
    assert ground(schema, out2, unambiguous) == []


def test_output_cap_is_passed_and_respected(tmp_path):
    long_reply = json.dumps({"text": "word " * 2000})
    cfg_cap = 64
    cfg, store, bstore, engine = make_env(tmp_path, responder=lambda p: long_reply, max_output_tokens=cfg_cap)
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.txt").write_text("hello world")
    task, _ = create(cfg, store, bstore, "Copy it.", [str(tmp_path / "in")], pace="full")
    run(cfg, store, bstore, engine, task)
    assert all(c["max_tokens"] == cfg_cap and c["emitted"] <= cfg_cap for c in engine.calls)


def test_preamble_is_short_and_every_item_uses_it(tmp_path):
    assert len(PREAMBLE.split()) < 80
    row, engine = job(tmp_path, good())
    assert all(PREAMBLE in c["prompt"] for c in engine.calls)
    assert "<<<FILE\n" + TEXT in engine.calls[0]["prompt"]


def test_schema_subset_is_enforced():
    with pytest.raises(SchemaError):
        parse_schema({"type": "object", "properties": {"x": {"type": "object"}}})
    with pytest.raises(SchemaError):
        parse_schema({"type": "array"})
    with pytest.raises(SchemaError):
        parse_schema({"type": "object", "properties": {"n": {"type": "number", "x-quote": True}}})


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": "b}"}\n```') == {"a": "b}"}
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


def test_array_quotes_keep_only_supported_items():
    schema = parse_schema({"type": "object", "properties": {"points": {"type": "array", "x-quote": True, "maxItems": 5}}})
    out = validate(schema, {"points": ["due date", "made up"], "points_quotes": ["Total due: $1,240.50", "never said"]})
    flags = ground(schema, out, TEXT)
    assert out["points"] == ["due date"] and flags == ["points"]
