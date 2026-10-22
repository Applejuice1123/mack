"""Output schemas and grounding checks for batch items.

The model is never trusted to stay grounded on its own. Every item's reply must
be one JSON object matching a small schema, and every field that claims to come
from the file is checked against the file mechanically. Anything that fails is
nulled and flagged for review, never shown as a result.

Supported schema subset (anything else is rejected at load with a clear message):
  object with properties of type string | number | integer | boolean,
  enum (a fixed list of strings), array of strings (optionally with enum),
  plus maxLength, maxItems, format: date (YYYY-MM-DD), description.
Extensions per property:
  "x-grounded": true   the value must appear in the file (case- and whitespace-
                       insensitive; dates match any written form of the same date)
  "x-quote": true      the model also returns <field>_quote (or <field>_quotes for
                       arrays), at most 25 words copied from the file, and the
                       quote must be found in the file
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

# The grounding rule, in every item's system prompt. Kept under 80 words.
PREAMBLE = (
    "You process one file at a time. Answer only from the text between <<<FILE and FILE>>>. "
    "If the text does not contain an answer, use null for that field. Never use outside knowledge, "
    "even if you know the answer. For a field with allowed options, pick exactly one of them or null. "
    "Reply with one JSON object and nothing else."
)

MAX_QUOTE_WORDS = 25
_SIMPLE = {"string", "number", "integer", "boolean"}


class SchemaError(ValueError):
    pass


@dataclass
class Field:
    name: str
    type: str  # string | number | integer | boolean | enum | array
    enum: Optional[list[str]] = None
    item_enum: Optional[list[str]] = None
    max_length: Optional[int] = None
    max_items: Optional[int] = None
    date: bool = False
    grounded: bool = False
    quote: bool = False
    description: str = ""


@dataclass
class Schema:
    fields: list[Field] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]


TEXT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}


def parse_schema(spec: dict, default_grounded: bool = False) -> Schema:
    if not isinstance(spec, dict) or spec.get("type") != "object" or not isinstance(spec.get("properties"), dict):
        raise SchemaError('schema must be {"type": "object", "properties": {...}}')
    fields = []
    for name, p in spec["properties"].items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise SchemaError(f"field name {name!r} must be a plain identifier")
        if not isinstance(p, dict):
            raise SchemaError(f"field {name!r} must be an object")
        t = p.get("type", "string" if "enum" in p else None)
        f = Field(name=name, type=t or "", description=p.get("description", ""))
        if "enum" in p:
            if not all(isinstance(e, str) for e in p["enum"]) or not p["enum"]:
                raise SchemaError(f"field {name!r}: enum must be a non-empty list of strings")
            f.type, f.enum = "enum", list(p["enum"])
        elif t == "array":
            items = p.get("items", {"type": "string"})
            if items.get("type", "string") != "string":
                raise SchemaError(f"field {name!r}: arrays may only hold strings")
            f.item_enum = list(items["enum"]) if "enum" in items else None
            f.max_items = p.get("maxItems")
        elif t not in _SIMPLE:
            raise SchemaError(f"field {name!r}: type {t!r} is not supported (string, number, integer, boolean, enum, array of strings)")
        f.max_length = p.get("maxLength")
        f.date = p.get("format") == "date"
        is_text = f.type == "string" or (f.type == "array" and not f.item_enum)
        f.grounded = bool(p.get("x-grounded", default_grounded and is_text and not p.get("x-quote")))
        f.quote = bool(p.get("x-quote", False))
        if f.quote and f.type not in ("string", "array"):
            raise SchemaError(f"field {name!r}: x-quote only applies to strings and arrays of strings")
        fields.append(f)
    if not fields:
        raise SchemaError("schema has no properties")
    return Schema(fields)


def load_schema(path: str | Path) -> Schema:
    try:
        spec = json.loads(Path(path).expanduser().read_text())
    except FileNotFoundError:
        raise SchemaError(f"schema file not found: {path}")
    except json.JSONDecodeError as e:
        raise SchemaError(f"schema file is not valid JSON: {e}")
    return parse_schema(spec)


# ---------------------------------------------------------------- prompting


def describe(schema: Schema) -> str:
    """The field list shown to the model."""
    lines = ["Reply with only a JSON object with exactly these fields:"]
    for f in schema.fields:
        if f.type == "enum":
            kind = "one of " + ", ".join(json.dumps(e) for e in f.enum) + ", or null"
        elif f.type == "array":
            opts = (" chosen from " + ", ".join(json.dumps(e) for e in f.item_enum)) if f.item_enum else ""
            kind = f"a list of {'up to ' + str(f.max_items) + ' ' if f.max_items else ''}strings{opts}, or []"
        elif f.date:
            kind = "a date as YYYY-MM-DD, or null"
        else:
            kind = {"string": "a string", "number": "a number", "integer": "a whole number", "boolean": "true or false"}[f.type] + ", or null"
        extra = []
        if f.max_length:
            extra.append(f"at most {f.max_length} characters")
        if f.grounded:
            extra.append("copied exactly as written in the text")
        if f.description:
            extra.append(f.description)
        lines.append(f'- "{f.name}": {kind}' + (f" ({'; '.join(extra)})" if extra else ""))
        if f.quote:
            key = f"{f.name}_quotes" if f.type == "array" else f"{f.name}_quote"
            what = "a list with one quote per item" if f.type == "array" else "a quote"
            lines.append(f'- "{key}": {what}: the exact words from the text (at most {MAX_QUOTE_WORDS}) that support it')
    return "\n".join(lines)


# ---------------------------------------------------------------- parsing


def extract_json(text: str) -> Any:
    """The first complete JSON object in a reply (code fences and chatter tolerated)."""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    start = text.find("{")
    if start == -1:
        raise ValueError("the reply contains no JSON object")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("the JSON object is not closed")


def validate(schema: Schema, obj: Any) -> dict:
    """Schema check. Returns the cleaned object (schema fields + quotes); raises ValueError."""
    if not isinstance(obj, dict):
        raise ValueError("the reply must be a JSON object")
    out: dict = {}
    for f in schema.fields:
        v = obj.get(f.name)
        if v is None:
            out[f.name] = [] if f.type == "array" else None
        elif f.type == "string":
            if not isinstance(v, str):
                raise ValueError(f'"{f.name}" must be a string')
            if f.max_length and len(v) > f.max_length:
                if f.grounded or f.quote or f.date:
                    raise ValueError(f'"{f.name}" is longer than {f.max_length} characters')
                # Free text that runs long (a wordy summary) is cut at a word boundary
                # rather than failing the whole item; nothing it claims is invented by cutting.
                cut = v[: f.max_length - 1].rsplit(" ", 1)[0].rstrip(",;: ")
                v = cut + "…"
                obj.setdefault("_trimmed", []).append(f.name)
            if f.date and not _iso_date(v):
                raise ValueError(f'"{f.name}" must be a date written as YYYY-MM-DD')
            out[f.name] = v
        elif f.type in ("number", "integer"):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f'"{f.name}" must be a number')
            if f.type == "integer" and float(v) != int(v):
                raise ValueError(f'"{f.name}" must be a whole number')
            out[f.name] = v
        elif f.type == "boolean":
            if not isinstance(v, bool):
                raise ValueError(f'"{f.name}" must be true or false')
            out[f.name] = v
        elif f.type == "enum":
            if v not in f.enum:
                raise ValueError(f'"{f.name}" must be one of {", ".join(f.enum)}, not {v!r}')
            out[f.name] = v
        elif f.type == "array":
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f'"{f.name}" must be a list of strings')
            if f.max_items is not None and len(v) > f.max_items:
                raise ValueError(f'"{f.name}" may have at most {f.max_items} items')
            if f.item_enum and any(x not in f.item_enum for x in v):
                bad = next(x for x in v if x not in f.item_enum)
                raise ValueError(f'"{f.name}" may only contain {", ".join(f.item_enum)}, not {bad!r}')
            out[f.name] = v
        if f.quote:
            key = f"{f.name}_quotes" if f.type == "array" else f"{f.name}_quote"
            q = obj.get(key)
            if f.type == "array":
                out[key] = [x if isinstance(x, str) else "" for x in q] if isinstance(q, list) else []
            else:
                out[key] = q if isinstance(q, str) else None
    if obj.get("_trimmed"):
        out["_trimmed"] = obj["_trimmed"]
    return out


# ---------------------------------------------------------------- grounding


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def appears(value: str, text_norm: str) -> bool:
    v = norm(value)
    return bool(v) and v in text_norm


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9


def _iso_date(v: str) -> Optional[date]:
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v.strip())
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def dates_in(text: str) -> dict[date, bool]:
    """Every date written in the text, mapped to whether it is unambiguous.

    "03/09/2025" could be 9 March or 3 September; both readings are returned as
    ambiguous. A value that only matches an ambiguous form cannot be verified, so
    grounding flags it for review rather than accepting it.
    """
    found: dict[date, bool] = {}

    def add(y, mo, d, sure=True):
        try:
            y = int(y)
            y = y + 2000 if y < 100 else y
            key = date(y, int(mo), int(d))
        except (ValueError, TypeError):
            return
        found[key] = found.get(key, False) or sure

    t = text.lower()
    for m in re.finditer(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", t):
        add(m[1], m[2], m[3])
    for m in re.finditer(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})\b", t):
        a, b = int(m[1]), int(m[2])
        ambiguous = a <= 12 and b <= 12 and a != b
        add(m[3], m[1], m[2], sure=not ambiguous)  # US order
        add(m[3], m[2], m[1], sure=not ambiguous)  # day-first order
    month = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
    for m in re.finditer(month + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", t):
        add(m[3], _MONTHS[m[1]], m[2])
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + month + r",?\s+(\d{4})", t):
        add(m[3], _MONTHS[m[2]], m[1])
    return found


def _number_appears(v: float, text_norm: str) -> bool:
    digits = re.sub(r"[,\s$€£]", "", text_norm)
    forms = {f"{v:g}", f"{v:.2f}"} | ({str(int(v))} if float(v) == int(v) else set())
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d])", digits) for f in forms)


def quote_ok(quote: Optional[str], text_norm: str) -> bool:
    return bool(quote) and len(quote.split()) <= MAX_QUOTE_WORDS and appears(quote.strip(" .\"'…"), text_norm)


def ground(schema: Schema, out: dict, text: str) -> list[str]:
    """Null what the file does not support. Mutates `out`; returns flagged field names."""
    tn = norm(text)
    flags = []
    text_dates = None
    for f in schema.fields:
        v = out.get(f.name)
        bad = False
        if f.grounded and v not in (None, [], ""):
            if f.type == "array":
                keep = [x for x in v if appears(x, tn)]
                bad = len(keep) != len(v)
                out[f.name] = keep
            elif f.type in ("number", "integer"):
                bad = not _number_appears(float(v), tn)
            elif f.date:
                text_dates = dates_in(text) if text_dates is None else text_dates
                d = _iso_date(v)
                if d in text_dates and not text_dates[d]:
                    # Written only as an ambiguous number like 03/09/2025: keep the value
                    # for review, but do not let it count as verified.
                    out.setdefault("_ambiguous", []).append(f.name)
                    flags.append(f.name)
                    continue
                bad = d not in text_dates
            elif isinstance(v, str):
                bad = not appears(v, tn)
            if bad and f.type != "array":
                out[f.name] = None
        if f.quote:
            if f.type == "array":
                key = f"{f.name}_quotes"
                items, quotes = out.get(f.name) or [], out.get(key) or []
                pairs = [(x, quotes[i] if i < len(quotes) else "") for i, x in enumerate(items)]
                keep = [(x, q) for x, q in pairs if quote_ok(q, tn)]
                if len(keep) != len(pairs):
                    bad = True
                out[f.name], out[key] = [x for x, _ in keep], [q for _, q in keep]
            elif out.get(f.name) not in (None, ""):
                key = f"{f.name}_quote"
                if not quote_ok(out.get(key), tn):
                    out[f.name], out[key], bad = None, None, True
        if bad:
            flags.append(f.name)
    return flags


def retention(original: str, rewritten: str) -> float:
    """Share of the original's distinct words that survive in a rewrite."""
    words = lambda s: set(re.findall(r"[a-z0-9']+", norm(s)))
    orig = {w for w in words(original) if len(w) > 2}
    return len(orig & words(rewritten)) / len(orig) if orig else 1.0
