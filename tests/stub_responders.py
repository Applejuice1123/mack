"""Reply functions for the stub engine, importable by subprocess tests via MACK_STUB_RESPONDER."""

import json
import re

from mack.stub import file_text


def invoice_fields(prompt: str) -> str:
    """For the rename recipe: read 'Vendor:' and 'Date:' lines the fixture files contain."""
    text = file_text(prompt)
    vendor = re.search(r"Vendor: (.+)", text)
    date = re.search(r"Date: (\d{4}-\d{2}-\d{2})", text)
    return json.dumps({
        "date": date.group(1) if date else None,
        "source": vendor.group(1).strip() if vendor else None,
        "kind": "invoice",
        "title": "Monthly invoice",
    })


def flag_note_002(prompt: str) -> str:
    """Grounded field: invents a value for note002, copies the first word otherwise."""
    text = file_text(prompt)
    if "Note 2 " in text:
        return json.dumps({"text": "Atlantis"})
    return json.dumps({"text": text.split()[0]})


def tagger(prompt: str) -> str:
    text = file_text(prompt)
    tag = "meeting" if "meeting" in text.lower() else "idea"
    return json.dumps({"tags": [tag], "summary": text.split(".")[0][:150]})
