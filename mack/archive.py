"""On-disk archive of every original message, searchable with SQLite FTS5.

Compaction never destroys anything: turns are written here when they happen, and
recall pulls the original text back when the user refers to it.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .textutil import STOPWORDS, content_words, entities, numbers, specials, split_sentences

PASSAGE_CHARS = 700

_CUES = re.compile(
    r"\b(?:earlier|before|previously|you said|i said|i told you|we (?:talked|discussed|decided|agreed)"
    r"|remind me|remember|recall|what was|what were|what did|which one did|go back to|again|last time"
    r"|at the (?:start|beginning)|originally|first message)\b",
    re.I,
)
_QUESTION = re.compile(
    r"\?\s*$|^\s*(?:what|which|who|whom|whose|when|where|why|how|did|do|does|is|are|was|were|can|could)\b", re.I
)
_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass
class Hit:
    turn: int
    part: int
    role: str
    text: str
    score: float


def passages(text: str, max_chars: int = PASSAGE_CHARS) -> list[str]:
    """Split a turn into passages at sentence boundaries so recall can be precise."""
    out, cur = [], ""
    for sent in split_sentences(text) or [text]:
        while len(sent) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:max_chars])
            sent = sent[max_chars:]
        if cur and len(cur) + len(sent) + 1 > max_chars:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def has_backreference(query: str) -> bool:
    return bool(_CUES.search(query))


def wants_recall(query: str) -> bool:
    """Recall costs context, so it only runs when the user is asking for something:
    an explicit reference to earlier, or a question."""
    return has_backreference(query) or bool(_QUESTION.search(query))


class Archive:
    def __init__(self, path: Path | str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5("
            "text, session UNINDEXED, turn UNINDEXED, part UNINDEXED, role UNINDEXED, ts UNINDEXED,"
            "tokenize='porter unicode61')"
        )
        self._db.commit()

    def add(self, session: str, turn: int, role: str, text: str) -> None:
        rows = [(p, session, turn, i, role, time.time()) for i, p in enumerate(passages(text))]
        with self._lock:
            self._db.executemany(
                "INSERT INTO passages(text, session, turn, part, role, ts) VALUES (?,?,?,?,?,?)", rows
            )
            self._db.commit()

    def search(self, session: str, query: str, before_turn: int, k: int = 3) -> list[Hit]:
        """Best passages from turns older than `before_turn` (i.e. no longer in the live context).

        Runs only for questions and explicit references to earlier. A hit must share
        something distinctive with the query: a number, a name, or two content words
        (one is enough when the user explicitly refers back).
        """
        terms = [t for t in _TERM.findall(query) if t.lower() not in STOPWORDS and len(t) > 1]
        if not terms or before_turn <= 0 or not wants_recall(query):
            return []
        match = " OR ".join('"' + t.replace('"', "") + '"' for t in dict.fromkeys(terms))
        with self._lock:
            rows = self._db.execute(
                "SELECT text, turn, part, role, bm25(passages) FROM passages "
                "WHERE passages MATCH ? AND session = ? AND CAST(turn AS INTEGER) < ? "
                "ORDER BY bm25(passages) LIMIT ?",
                (match, session, before_turn, k * 4),
            ).fetchall()
        q_words, q_nums, q_names = content_words(query), numbers(query), {e.lower() for e in entities(query)}
        need = 1 if has_backreference(query) else 2
        hits = []
        for text, turn, part, role, bm25 in rows:
            low = text.lower()
            distinctive = bool(q_nums & numbers(text)) or any(n in low for n in q_names)
            overlap = len(q_words & content_words(text))
            if not (distinctive or overlap >= need):
                continue
            # What the user said, and passages that carry a concrete detail, are the
            # likelier target than an acknowledgement that repeats the same words.
            score = -bm25 * (1.3 if role == "user" else 1.0) * (1.3 if numbers(text) or specials(text) else 1.0)
            hits.append(Hit(int(turn), int(part), role, text, score))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def turn_text(self, session: str, turn: int) -> str:
        with self._lock:
            rows = self._db.execute(
                "SELECT text FROM passages WHERE session = ? AND CAST(turn AS INTEGER) = ? "
                "ORDER BY CAST(part AS INTEGER)",
                (session, turn),
            ).fetchall()
        return " ".join(r[0] for r in rows)

    def delete_session(self, session: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM passages WHERE session = ?", (session,))
            self._db.commit()

    def close(self) -> None:
        self._db.close()
