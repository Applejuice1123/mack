"""Assembles engine, governor, archive and sessions. Shared by the CLI and the server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .archive import Archive
from .config import Config
from .engine import LazyEngine
from .governor import SPEC_MIN_SPEEDUP, Governor, load_json, set_thread_user_initiated
from .explain import explain
from .session import Session


def draft_wanted(cfg: Config) -> bool:
    """Only pay for the draft model's memory if speculative decoding will be used."""
    mode = cfg.generation.speculative
    if mode == "on":
        return True
    if mode != "auto" or not cfg.model.draft:
        return False
    phase0 = load_json(cfg.state_dir / "phase0.json", cfg.model.main) or {}
    return (phase0.get("speculative", {}).get("speedup") or 0.0) >= SPEC_MIN_SPEEDUP


class Mack:
    """Must be constructed and used on a single thread: MLX work stays on one stream."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_thread_user_initiated()
        self.engine = LazyEngine(cfg, load_draft=draft_wanted(cfg))
        self.governor = Governor(cfg, spec_supported=self.engine.spec_supported)
        self.governor.start()
        self.archive = Archive(cfg.state_dir / "archive.db")
        self.sessions_dir = cfg.state_dir / "sessions"
        self.sessions: dict[str, Session] = {}

    @property
    def budget(self) -> int:
        """The configured budget, lowered to the measured memory ceiling if Phase 0 ran."""
        budget = self.cfg.context.budget_tokens
        ceiling = (self.governor.phase0 or {}).get("memory", {}).get("context_ceiling_tokens")
        return min(budget, ceiling) if ceiling else budget

    def new_session(self) -> Session:
        s = Session(self.engine, self.cfg, self.archive, self.governor, budget_tokens=self.budget)
        self.sessions[s.id] = s
        return s

    def session(self, session_id: Optional[str]) -> Session:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        path = self.sessions_dir / f"{session_id}.json" if session_id else None
        if path and path.exists():
            s = Session.load(path, self.engine, self.cfg, self.archive, self.governor)
            self.sessions[s.id] = s
            return s
        return self.new_session()

    def saved_sessions(self) -> list[str]:
        if not self.sessions_dir.exists():
            return []
        files = sorted(self.sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files]

    def unload_engine(self) -> None:
        """Give the model's memory back. Chat sessions are saved after every turn and
        rebuild their context from disk on next use."""
        self.sessions.clear()
        self.engine.unload()

    def save(self, session: Session) -> Path:
        return session.save(self.sessions_dir)

    def explain(self, text: str, style: str = "explain", question: Optional[str] = None, cancel=None):
        """One-shot explanation of captured screen text, under the same governor as chat."""
        for event in explain(self.engine, text, style, question, policy=self.governor.policy(), cancel=cancel):
            if event["type"] == "done":
                self.governor.last_tps = event["tps"]
            yield event

    def status(self, session: Optional[Session] = None) -> dict:
        return {
            "features": ["chat", "explain", "tasks", "batch"],
            "model": self.cfg.model.main,
            "engine_loaded": self.engine.loaded,
            "adapter": self.engine.adapter_loaded if self.engine.loaded else None,
            "hybrid": self.engine.hybrid if self.engine.loaded else None,
            "memory": self.engine.memory() if self.engine.loaded else None,
            "governor": self.governor.status(),
            "session": session.status() if session else None,
        }
