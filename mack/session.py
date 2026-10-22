"""A conversation: live context, compaction, recall.

Eviction here is compaction, not cache surgery. On a hybrid model the recurrent
layers still "remember" tokens whose attention entries were dropped, which is a
state the model never saw in training. So when the context fills, Mack throws
the whole cache away and prefills a rebuilt one:

    system prompt
    + Memory: exact-quote pins (numbers, names, decisions) + a verified summary
    + the most recent turns, verbatim

Rebuilding costs a full prefill, so it happens rarely and frees a lot each time.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from .archive import Archive, wants_recall
from .config import Config
from .pins import Pin, extract_pins, select_pins
from .prompts import SUMMARY_INSTRUCTION, SYSTEM_PROMPT, memory_block, recall_block
from .textutil import clip_middle, content_words, split_sentences
from .verify import verify_summary

SUMMARY_CHUNK_TOKENS = 2000
SUMMARY_MAX_BULLETS = 8
SUMMARY_MAX_TOKENS = 220
MIN_REPLY_TOKENS = 128


@dataclass
class Turn:
    idx: int
    role: str
    text: str
    n_tokens: int


class Session:
    def __init__(
        self,
        engine,
        cfg: Config,
        archive: Archive,
        governor=None,
        session_id: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        budget_tokens: Optional[int] = None,
    ):
        self.engine = engine
        self.cfg = cfg
        self.archive = archive
        self.governor = governor
        self.id = session_id or uuid.uuid4().hex[:12]
        self.system_prompt = system_prompt
        self.budget = budget_tokens or cfg.context.budget_tokens
        self.fmt = engine.fmt
        self.turns: list[Turn] = []
        self.live_from = 0  # first turn that is verbatim in the live context
        self.pins: list[Pin] = []
        self.summary: list[str] = []
        self.compactions: list[dict] = []
        self.ctx = None
        self._rebuild()

    # ---- rendering ---------------------------------------------------------

    def _system_text(self) -> str:
        memory = memory_block(self.pins, self.summary)
        return f"{self.system_prompt}\n\n{memory}" if memory else self.system_prompt

    def _live_text(self, turn: Turn, max_chars: Optional[int] = None) -> str:
        return clip_middle(turn.text, max_chars) if max_chars else turn.text

    def _render_context(self) -> str:
        parts = [self.fmt.system(self._system_text())]
        # A single huge old message must not eat the recent-turn budget on its own.
        clip = int(self.budget * self.cfg.context.recent_fraction * 0.5 * 3.5)
        for turn in self.turns[self.live_from :]:
            text = self._live_text(turn, clip)
            parts.append(self.fmt.user(text) if turn.role == "user" else self.fmt.assistant(text))
        return "".join(parts)

    def _rebuild(self, progress: Optional[Callable[[int, int], None]] = None) -> None:
        self.ctx = None  # free the old cache before allocating the new one
        self._recalled: set[tuple[int, int]] = set()  # recall blocks do not survive a rebuild
        ctx = self.engine.new_context()
        self.engine.prefill(ctx, self.engine.encode(self._render_context()), progress)
        self.ctx = ctx

    # ---- turn --------------------------------------------------------------

    def _policy(self):
        return self.governor.policy() if self.governor else None

    def _recall(self, text: str, budget: int) -> list:
        if not self.cfg.recall.enabled or self.live_from == 0 or budget <= 0:
            return []
        hits = self.archive.search(self.id, text, before_turn=self.live_from, k=self.cfg.recall.top_k)
        pinned = {p.quote for p in self.pins}
        query = content_words(text)
        kept, used = [], 0
        for h in hits:
            # Skip what the model can already see: an earlier recall still in the live
            # context, or a passage whose relevant sentences are all pinned in Memory.
            relevant = [x for x in split_sentences(h.text) if query & content_words(x)]
            if (h.turn, h.part) in self._recalled or all(x in pinned for x in relevant):
                continue
            cost = self.engine.count(h.text) + 12
            if used + cost > budget:
                break
            kept.append(h)
            used += cost
        self._recalled.update((h.turn, h.part) for h in kept)
        return kept

    def chat(
        self,
        user_text: str,
        max_tokens: Optional[int] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> Iterator[dict]:
        """Run one turn. Yields events: status, token, done."""
        from .engine import GenStats, strip_thinking

        max_tokens = max_tokens or self.cfg.generation.max_tokens
        user_text = user_text.strip()
        user_tokens = self.engine.count(user_text)
        user = Turn(len(self.turns), "user", user_text, user_tokens)
        overhead = self.engine.count(self.fmt.user("") + self.fmt.generation_prompt()) + 8
        recall_cfg = self.cfg.recall
        may_recall = recall_cfg.enabled and wants_recall(user_text)
        recall_max = min(recall_cfg.max_tokens, self.budget // 8)

        # Compact first, so recall then searches everything that just left the window
        # and is budgeted against the room that actually remains.
        need = user_tokens + overhead + (recall_max if may_recall else 0)
        used = len(self.ctx.tokens)
        over_high_water = used + need > self.cfg.context.high_water * self.budget
        if (over_high_water or used + need + max_tokens > self.budget) and self._can_compact():
            yield {"type": "status", "phase": "compact_start", "context_tokens": used}
            record = self._compact()
            yield {"type": "status", "phase": "compact_done", **record}
            used = len(self.ctx.tokens)

        room = self.budget - used - MIN_REPLY_TOKENS - overhead
        live_user = user_text
        if user_tokens > room:
            # The message alone does not fit. Its full text goes to the archive; the
            # live context gets the head and tail.
            live_user = clip_middle(user_text, int(len(user_text) * max(room, 64) / user_tokens * 0.9))
            user_tokens = self.engine.count(live_user)
            yield {"type": "status", "phase": "clipped", "kept_tokens": user_tokens}

        # Recalled text is never clipped: it gets only the room left after the message.
        hits = self._recall(user_text, min(recall_max, room - user_tokens)) if may_recall else []
        if hits:
            live_user = f"{recall_block(hits)}\n\n{live_user}"
            yield {"type": "status", "phase": "recall", "turns": sorted({h.turn + 1 for h in hits})}

        delta = self.engine.encode(self.fmt.user(live_user) + self.fmt.generation_prompt())
        max_tokens = max(MIN_REPLY_TOKENS, min(max_tokens, self.budget - used - len(delta)))

        self.turns.append(user)
        self.archive.add(self.id, user.idx, "user", user_text)

        stats = GenStats()
        detok = self.engine.detok()
        pieces: list[str] = []
        t0 = time.perf_counter()
        ttft = None
        try:
            for tok in self.engine.generate(
                self.ctx, delta, max_tokens, policy=self._policy(), cancel=cancel, stats=stats
            ):
                if ttft is None:
                    ttft = time.perf_counter() - t0  # includes prefill of the new turn
                seg = detok.add(tok)
                if seg:
                    pieces.append(seg)
                    yield {"type": "token", "text": seg}
            seg = detok.flush()
            if seg:
                pieces.append(seg)
                yield {"type": "token", "text": seg}
        finally:
            # Runs even if the consumer walks away mid-reply: the cache has consumed the
            # partial reply, so the turn list must record it too.
            reply = strip_thinking("".join(pieces)).strip()
            assistant = Turn(len(self.turns), "assistant", reply, stats.n_tokens)
            self.turns.append(assistant)
            self.archive.add(self.id, assistant.idx, "assistant", reply)
        if self.governor:
            self.governor.observe(stats)
        yield {
            "type": "done",
            "reply": reply,
            "ttft_s": round(ttft or 0.0, 3),
            "recalled": sorted({h.turn + 1 for h in hits}),
            **stats.as_dict(),
            **self.status(),
        }

    # ---- compaction --------------------------------------------------------

    def _plan_live_from(self) -> int:
        """Earliest turn that still fits the verbatim window, on a user-turn boundary,
        always keeping at least the last exchange."""
        budget = int(self.budget * self.cfg.context.recent_fraction)
        used = 0
        start = len(self.turns)
        for turn in reversed(self.turns):
            if used + turn.n_tokens > budget and len(self.turns) - start >= 2:
                break
            used += turn.n_tokens
            start = turn.idx
        while start < len(self.turns) and self.turns[start].role != "user":
            start += 1
        return max(start, self.live_from)

    def _can_compact(self) -> bool:
        return self._plan_live_from() > self.live_from

    def _summarize(self, turns: list[Turn]) -> tuple[list[str], int]:
        """Model-written bullets for `turns`, each checked against its source chunk.
        Returns (kept bullets, number dropped)."""
        chunks, cur, cur_tokens = [], [], 0
        for turn in turns:
            who = "User" if turn.role == "user" else "Mack"
            line = f"{who}: {turn.text}"
            if cur and cur_tokens + turn.n_tokens > SUMMARY_CHUNK_TOKENS:
                chunks.append("\n".join(cur))
                cur, cur_tokens = [], 0
            cur.append(clip_middle(line, SUMMARY_CHUNK_TOKENS * 3))
            cur_tokens += turn.n_tokens
        if cur:
            chunks.append("\n".join(cur))

        kept, dropped = [], 0
        judge = self.engine if self.cfg.context.verify_with_model else None
        for chunk in chunks:
            prompt = SUMMARY_INSTRUCTION.format(max_bullets=SUMMARY_MAX_BULLETS, excerpt=chunk)
            raw = self.engine.complete(
                [{"role": "user", "content": prompt}], max_tokens=SUMMARY_MAX_TOKENS, policy=self._policy()
            )
            for verdict in verify_summary(raw, chunk, engine=judge):
                if verdict.kept:
                    kept.append(verdict.line)
                else:
                    dropped += 1
        return kept, dropped

    def _compact(self) -> dict:
        t0 = time.perf_counter()
        before = len(self.ctx.tokens)
        new_from = self._plan_live_from()
        leaving = self.turns[self.live_from : new_from]
        self.ctx = None  # the summarizer needs the memory the old cache was holding

        ctxcfg = self.cfg.context
        self.pins = select_pins(
            self.pins + extract_pins(leaving), int(self.budget * ctxcfg.pin_fraction), self.engine.count
        )
        bullets, dropped = self._summarize(leaving)
        pinned_words = [content_words(p.quote) for p in self.pins]
        for b in bullets:
            words = content_words(b)
            if any(words and len(words & pw) / len(words) >= 0.8 for pw in pinned_words):
                continue  # already kept as an exact quote
            if b not in self.summary:
                self.summary.append(b)
        # Oldest bullets fall off first. They are not re-summarized: a summary of a
        # summary drifts, and the originals are still in the archive.
        limit = int(self.budget * ctxcfg.summary_fraction)
        while self.summary and sum(self.engine.count(s) + 3 for s in self.summary) > limit:
            self.summary.pop(0)

        self.live_from = new_from
        self._rebuild()
        record = {
            "at_turn": len(self.turns),
            "tokens_before": before,
            "tokens_after": len(self.ctx.tokens),
            "turns_compacted": len(leaving),
            "pins": len(self.pins),
            "summary_bullets": len(self.summary),
            "summary_dropped": dropped,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        self.compactions.append(record)
        return record

    # ---- state -------------------------------------------------------------

    def status(self) -> dict:
        return {
            "session": self.id,
            "context_tokens": len(self.ctx.tokens) if self.ctx else 0,
            "budget_tokens": self.budget,
            "turns": len(self.turns),
            "live_from_turn": self.live_from + 1,
            "pins": len(self.pins),
            "summary_bullets": len(self.summary),
            "compactions": len(self.compactions),
            "cache_mb": round(self.engine.cache_bytes(self.ctx) / 1e6, 1) if self.ctx else 0.0,
        }

    def memory_view(self) -> dict:
        return {
            "pins": [p.as_dict() for p in self.pins],
            "summary": list(self.summary),
            "compactions": list(self.compactions),
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.id}.json"
        path.write_text(
            json.dumps(
                {
                    "id": self.id,
                    "system_prompt": self.system_prompt,
                    "budget": self.budget,
                    "live_from": self.live_from,
                    "turns": [asdict(t) for t in self.turns],
                    "pins": [p.as_dict() for p in self.pins],
                    "summary": self.summary,
                    "compactions": self.compactions,
                }
            )
        )
        return path

    @classmethod
    def load(cls, path: Path, engine, cfg: Config, archive: Archive, governor=None) -> "Session":
        data = json.loads(Path(path).read_text())
        self = cls.__new__(cls)
        self.engine, self.cfg, self.archive, self.governor = engine, cfg, archive, governor
        self.id = data["id"]
        self.system_prompt = data["system_prompt"]
        self.budget = data["budget"]
        self.fmt = engine.fmt
        self.turns = [Turn(**t) for t in data["turns"]]
        self.live_from = data["live_from"]
        self.pins = [Pin(**p) for p in data["pins"]]
        self.summary = data["summary"]
        self.compactions = data["compactions"]
        self.ctx = None
        self._rebuild()
        return self
