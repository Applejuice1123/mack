"""Deterministic fixture sets for the model tests and the demo.

    python tests/fixtures/make_fixtures.py vault DIR [--n 300]   markdown notes, five kinds, 100-2,000 words
    python tests/fixtures/make_fixtures.py invoices DIR [--n 30] invoice-like text files, with an answers.json
    python tests/fixtures/make_fixtures.py inbox DIR             8 invoice PDFs, 6 screenshots, 4 notes, 2 blank images
    python tests/fixtures/make_fixtures.py mixed DIR             the 40-file set for the recipe model test
    python tests/fixtures/make_fixtures.py entities DIR          20 notes naming famous entities without facts

PDFs and images are drawn by the Mack app's helper (app/build.sh first).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "app" / "Mack.app" / "Contents" / "MacOS" / "Mack"

VENDORS = ["Northwind Traders", "Contoso Ltd", "Fabrikam Inc", "Blue Yonder Airlines", "Tailspin Toys", "Litware Systems",
           "Adventure Works", "Wide World Importers", "Proseware Labs", "Lucerne Publishing"]
PEOPLE = ["Priya", "Tomas", "Ingrid", "Kofi", "Yuki", "Mateo", "Leila", "Anders"]
TOPICS = ["the storage migration", "the onboarding flow", "quarterly planning", "the search rewrite", "hiring",
          "the mobile release", "billing errors", "the design system"]
SENTENCES = [
    "We spent most of the time on {t}.", "{p} raised a concern about {t}.", "The team agreed to revisit {t} next week.",
    "Nobody had strong objections, but {p} wants more data.", "There is a dependency on {t} that we keep forgetting.",
    "{p} will write up the options before Friday.", "The numbers looked better than last month.",
    "We should not ship before the load test passes.", "This connects to an older note about {t}.",
]
KINDS = {
    "meeting": "# {t} sync\n\nAttendees: {p}, {q}\n\n",
    "idea": "# Idea: rethink {t}\n\n",
    "bug": "# Bug: crash in {t}\n\nSteps to reproduce:\n1. Open the app\n2. Go to {t}\n\n",
    "reference": "# Reference: how {t} works\n\n",
    "personal": "# Weekend\n\nWent for a long walk and thought about {t}.\n\n",
}


def words_of(r: random.Random, target: int, t: str) -> str:
    out = []
    while sum(len(s.split()) for s in out) < target:
        out.append(r.choice(SENTENCES).format(t=t, p=r.choice(PEOPLE)))
        if len(out) % 6 == 0:
            out.append("\n\n")
    return " ".join(out).replace(" \n\n ", "\n\n")


def vault(folder: Path, n: int = 300, seed: int = 7) -> None:
    r = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        kind = list(KINDS)[i % 5]
        t = r.choice(TOPICS)
        head = KINDS[kind].format(t=t, p=r.choice(PEOPLE), q=r.choice(PEOPLE))
        body = words_of(r, r.choice([100, 200, 400, 800, 2000]) if i % 10 else 100, t)
        front = f"---\ncreated: {date(2025, 1, 1) + timedelta(days=i)}\n---\n" if i % 3 == 0 else ""
        (folder / f"{i:04d}-{kind}.md").write_text(front + head + body + "\n")


def invoice_text(r: random.Random, i: int) -> tuple[str, dict]:
    vendor = VENDORS[i % len(VENDORS)]
    d = date(2025, 6, 1) + timedelta(days=r.randint(0, 400))
    amount = round(r.uniform(20, 5000), 2)
    style = i % 3
    written = d.isoformat() if style == 0 else (d.strftime("%B %-d, %Y") if style == 1 else d.strftime("%d/%m/%Y"))
    text = (
        f"{vendor}\n{r.randint(10, 999)} Market Street\n\nINVOICE #{1000 + i}\nInvoice date: {written}\n\n"
        f"Description                 Qty   Price\nConsulting services          {r.randint(1, 9)}   ${amount / 2:,.2f}\n"
        f"Support plan                 1     ${amount / 2:,.2f}\n\nTotal due: ${amount:,.2f}\nPayment terms: net 30\n"
    )
    return text, {"vendor": vendor, "date": d.isoformat(), "amount": amount}


def invoices(folder: Path, n: int = 30, seed: int = 11) -> None:
    r = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    answers = {}
    for i in range(n):
        text, truth = invoice_text(r, i)
        name = f"invoice-{i:02d}.txt"
        (folder / name).write_text(text)
        answers[name] = truth
    (folder.parent / f"{folder.name}-answers.json").write_text(json.dumps(answers, indent=2))


def _helper(*args) -> None:
    if not HELPER.exists():
        raise SystemExit("build the Mack app first: app/build.sh")
    subprocess.run([str(HELPER), *args], check=True)


def inbox(folder: Path, seed: int = 13) -> None:
    r = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    answers = {}
    for i in range(8):
        text, truth = invoice_text(r, i)
        name = f"scan{i:02d}.pdf"
        _helper("--make-pdf", str(folder / name), text, *(["--scan"] if i % 2 else []))
        answers[name] = truth
    shots = ["Order confirmed\nAdventure Works\nShipped on 2026-02-14", "Meeting moved\nThursday 3pm, room 4",
             "Your receipt from Tailspin Toys\nMarch 9, 2026\nTotal $42.10", "Error 503\nService unavailable",
             "Flight BY 221\nBlue Yonder Airlines\nDeparts 2026-07-01 08:15", "Password changed\nLitware Systems account"]
    for i, s in enumerate(shots):
        _helper("--make-image", str(folder / f"screenshot{i}.png"), s)
    for i in range(4):
        (folder / f"note{i}.md").write_text(f"# Note {i}\n\nRemember to call {r.choice(PEOPLE)} about {r.choice(TOPICS)}.\n")
    for i in range(2):
        _helper("--make-image", str(folder / f"blank{i}.png"), " ")
    (folder.parent / f"{folder.name}-answers.json").write_text(json.dumps(answers, indent=2))


def mixed(folder: Path) -> None:
    vault(folder / "notes", 15, seed=21)
    inbox_dir = folder / "docs"
    inbox(inbox_dir)
    r = random.Random(5)
    t = folder / "transcripts"
    t.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        lines = []
        for _ in range(12):
            s = r.choice(SENTENCES).format(t=r.choice(TOPICS), p=r.choice(PEOPLE))
            lines.append(r.choice(["um, ", "so, like, ", "uh ", "you know, ", ""]) + s[0].lower() + s[1:] + r.choice(["", " um", " you know"]))
        lines.append(f"okay so {r.choice(PEOPLE)} will send the notes by friday")
        (t / f"call{i}.txt").write_text("\n".join(lines) + "\n")


ENTITIES = [("France", "capital"), ("Japan", "capital"), ("Brazil", "capital"), ("Canada", "capital"), ("Egypt", "capital"),
            ("Microsoft", "founder"), ("Apple", "founder"), ("Tesla", "founder"), ("Amazon", "founder"), ("Nintendo", "founder"),
            ("Marie Curie", "birth_year"), ("Albert Einstein", "birth_year"), ("Ada Lovelace", "birth_year"),
            ("Isaac Newton", "birth_year"), ("Frida Kahlo", "birth_year"), ("Germany", "capital"), ("Kenya", "capital"),
            ("Sony", "founder"), ("Nikola Tesla", "birth_year"), ("Australia", "capital")]


def entities(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, (name, field) in enumerate(ENTITIES):
        (folder / f"e{i:02d}.md").write_text(f"# Reading list\n\nI keep meaning to read more about {name}. Maybe this weekend.\n")
    (folder.parent / f"{folder.name}-fields.json").write_text(json.dumps({f"e{i:02d}.md": f for i, (_, f) in enumerate(ENTITIES)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["vault", "invoices", "inbox", "mixed", "entities"])
    ap.add_argument("dir")
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    {"vault": lambda: vault(d, a.n or 300), "invoices": lambda: invoices(d, a.n or 30), "inbox": lambda: inbox(d),
     "mixed": lambda: mixed(d), "entities": lambda: entities(d)}[a.kind]()
    print(f"wrote {a.kind} fixtures to {d}")
