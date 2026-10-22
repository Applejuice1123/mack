"""Small text helpers shared by pin extraction, summary verification and recall."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if then else so because as of at by for with about against between into
    through during before after above below to from up down in out on off over under again further
    once here there when where why how all any both each few more most other some such no nor not
    only own same than too very can will just should now is are was were be been being have has had
    having do does did doing would could i me my myself we our ours you your yours he him his she her
    it its they them their what which who whom this that these those am s t d ll m re ve y also
    said say says tell told ask asked asks want wanted wants need needed needs like get got make made
    thing things something anything one two user mack assistant please thanks thank okay ok yes yeah
    earlier before previously mentioned remind remember recall again""".split()
)

_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?(?![\w])|\b\d[\d,]*(?:\.\d+)?(?=[a-zA-Z%])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
_SPECIAL = re.compile(
    r"https?://\S+|[\w.+-]+@[\w-]+\.[\w.-]+|(?:~|\.{1,2})?/(?:[\w.-]+/)+[\w.-]+|\b(?:[\w-]+/)+[\w-]+\.\w+\b"
    r"|\b\w+\.(?:py|js|ts|md|json|toml|c|cpp|h|txt|csv|pdf)\b"
    r"|\b[A-Za-z]+_[A-Za-z0-9_]+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
)
_CAP = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:[-'][A-Za-z0-9]+)*\b")


def split_sentences(text: str) -> list[str]:
    """Sentence-ish units. Lines are hard boundaries (bullets, code, short statements)."""
    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped:
            continue
        stripped = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", stripped)
        for part in _SENT_END.split(stripped):
            part = part.strip()
            if part:
                out.append(part)
    return out


def numbers(text: str) -> set[str]:
    """Numeric literals, normalized so 1,200 == 1200 and $40 == 40."""
    found = set()
    for m in _NUMBER.finditer(text):
        n = m.group(0).lstrip("+-$").replace(",", "").rstrip("%")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        if n:
            found.add(n)
    return found


def specials(text: str) -> set[str]:
    """URLs, emails, paths, filenames, identifiers: things that must survive verbatim."""
    return {m.group(0).rstrip(".,;:)") for m in _SPECIAL.finditer(text)}


def entities(text: str) -> set[str]:
    """Capitalized names, skipping words that are capitalized only for starting a sentence."""
    found = set()
    for sent in split_sentences(text) or [text]:
        caps = list(_CAP.finditer(sent))
        for m in caps:
            word = m.group(0)
            first = m.start() == 0 or not sent[: m.start()].strip(" \"'([*_#>-")
            if word.lower() in STOPWORDS:
                continue
            # A sentence-initial word counts only if it also looks like a name elsewhere
            # (all caps, digits or internal capitals), e.g. "NASA", "GPT4", "MacBook".
            if first and not (word.isupper() and len(word) > 1 or any(c.isdigit() for c in word) or re.search(r"[a-z][A-Z]", word)):
                continue
            if len(word) == 1 and word != "I":
                continue
            if word == "I":
                continue
            found.add(word)
    return found


def stem(word: str) -> str:
    w = word.lower().strip("'-")
    if w.endswith("'s"):
        w = w[:-2]
    for suffix in ("ingly", "edly", "ing", "ed", "es", "ly", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def content_words(text: str) -> set[str]:
    return {stem(w) for w in _WORD.findall(text) if w.lower() not in STOPWORDS and len(w) > 2}


def clip_middle(text: str, max_chars: int, marker: str = "\n[… middle of this message is archived …]\n") -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(marker))
    head = keep * 2 // 3
    return text[:head] + marker + text[len(text) - (keep - head) :]
