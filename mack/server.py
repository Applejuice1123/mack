"""HTTP server: task dashboard, chat, explain, and a small JSON/SSE API.

All MLX work runs on one worker thread that owns the engine. HTTP handlers only
pass messages to it, so a slow client can never stall or share the GPU stream.
Background tasks run on that same thread in small steps; an interactive request
is served between two steps, so it waits at most a moment.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .app import Mack
from .config import Config
from .tasks import TaskScheduler, TaskStore

WEB = Path(__file__).resolve().parent / "web"
_END = object()


class ChatRequest(BaseModel):
    message: str
    session: Optional[str] = None
    max_tokens: Optional[int] = None


class ExplainRequest(BaseModel):
    text: str
    style: str = "explain"
    question: Optional[str] = None


class TaskRequest(BaseModel):
    prompt: str
    kind: str = "auto"
    pages: int = 5
    pace: str = "auto"
    out_dir: Optional[str] = None


class Worker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.jobs: queue.Queue = queue.Queue()
        self.scheduler: Optional[TaskScheduler] = None
        self.ready = threading.Event()
        self.error: Optional[str] = None
        self.mack: Optional[Mack] = None
        self.cancel = threading.Event()  # cancel flag of the request being served
        threading.Thread(target=self._run, name="mack-engine", daemon=True).start()

    def _run(self) -> None:
        try:
            self.mack = Mack(self.cfg)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.ready.set()
            return
        self.scheduler = TaskScheduler(self.mack.engine, self.cfg, TaskStore(self.cfg), governor=self.mack.governor)
        self.ready.set()
        last_work = time.monotonic()
        while True:
            try:
                wait = self.scheduler.tick()
            except Exception as e:  # a broken task must not take the engine down with it
                print(f"[tasks] {type(e).__name__}: {e}")
                wait = 5.0
            now = time.monotonic()
            if wait is not None:
                last_work = now
            elif self.mack.engine.loaded and now - last_work > self.cfg.background.unload_after_idle_s:
                self.mack.unload_engine()  # nothing to do: give the ~3 GB back until needed
            try:
                if wait == 0:
                    fn, out = self.jobs.get_nowait()
                else:
                    fn, out = self.jobs.get(timeout=2.0 if wait is None else wait)
            except queue.Empty:
                continue
            try:
                fn(out)
            except Exception as e:
                out.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                out.put(_END)
                last_work = time.monotonic()

    def submit(self, fn) -> queue.Queue:
        out: queue.Queue = queue.Queue()
        self.jobs.put((fn, out))
        return out

    def call(self, fn):
        """Run fn(mack) on the engine thread and return its result."""
        out = self.submit(lambda q: q.put(fn(self.mack)))
        result = out.get()
        if isinstance(result, dict) and result.get("type") == "error":
            raise RuntimeError(result["message"])
        return result


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="Mack.ai")
    worker = Worker(cfg)

    def not_ready():
        if not worker.ready.is_set():
            return JSONResponse({"loading": True}, status_code=503)
        if worker.error:
            return JSONResponse({"error": worker.error}, status_code=500)
        return None

    store = TaskStore(cfg)

    @app.get("/")
    def index():
        return FileResponse(WEB / "tasks.html", headers={"Cache-Control": "no-store"})

    @app.get("/chat")
    def chat_page():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/paces")
    def paces():
        from .tasks import PACES, pace_rate

        return {name: {"tps": pace_rate(cfg, name), "cpu_slowdown_target": PACES[name][0]} for name in PACES}

    @app.get("/api/tasks")
    def list_tasks():
        tasks = [t.summary() for t in reversed(store.all())]
        return {"tasks": tasks, "engine_ready": worker.ready.is_set() and not worker.error}

    @app.post("/api/tasks")
    def add_task(req: TaskRequest):
        try:
            task = store.create(req.prompt, req.kind, req.pages, req.pace, req.out_dir)
        except (ValueError, OSError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return task.summary()

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: str):
        try:
            task = store.load(task_id)
        except FileNotFoundError:
            return JSONResponse({"error": "unknown task"}, status_code=404)
        if task.kind == "batch":
            from .batch import BatchStore

            bstore = BatchStore(cfg)
            recent = []
            for r in bstore.recent(task.id, 40):
                recent.append({"path": r["path"], "status": r["status"], "output": json.loads(r["output"]) if r["output"] else None,
                               "flags": json.loads(r["flags"]) if r["flags"] else [], "reason": r["reason"] or r["error"]})
            return {**task.summary(), "items": recent, "stats": bstore.stats(task.id), "log": task.log[-30:],
                    "live": store.live(task_id)[-4000:], "units": [], "files": [task.batch.get("output_path", "")]}
        out = Path(task.out_dir)
        files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()) if out.exists() else []
        return {
            **task.summary(),
            "units": [
                {"title": u.title, "spec": u.spec, "status": u.status, "words": u.words, "notes": u.notes}
                for u in task.units
            ],
            "log": task.log[-30:],
            "live": store.live(task_id)[-6000:],
            "files": files[:200],
        }

    @app.post("/api/tasks/{task_id}/{action}")
    def control_task(task_id: str, action: str):
        try:
            if action == "open":  # reveal the output folder in Finder
                import subprocess

                out = Path(store.load(task_id).out_dir)
                if out.exists():
                    subprocess.run(["open", str(out)], check=False)
                return {"ok": out.exists()}
            store.set_control(task_id, action)
        except FileNotFoundError:
            return JSONResponse({"error": "unknown task"}, status_code=404)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return store.load(task_id).summary()

    @app.get("/api/status")
    def status(session: Optional[str] = None):
        if (r := not_ready()) is not None:
            return r
        mack = worker.mack
        s = mack.sessions.get(session) if session else None
        return {**mack.status(s), "task": worker.scheduler.active() if worker.scheduler else None}

    @app.get("/api/memory")
    def memory(session: str):
        if (r := not_ready()) is not None:
            return r
        s = worker.mack.sessions.get(session)
        return s.memory_view() if s else JSONResponse({"error": "unknown session"}, status_code=404)

    @app.get("/api/sessions")
    def sessions():
        if (r := not_ready()) is not None:
            return r
        return {"saved": worker.mack.saved_sessions(), "open": list(worker.mack.sessions)}

    @app.post("/api/session")
    def new_session():
        if (r := not_ready()) is not None:
            return r
        return worker.call(lambda m: m.new_session().status())

    @app.post("/api/cancel")
    def cancel():
        worker.cancel.set()
        return {"ok": True}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if (r := not_ready()) is not None:
            return r

        cancelled = threading.Event()  # per request, so a late disconnect cannot hit the next one

        def job(out: queue.Queue):
            mack = worker.mack
            session = mack.session(req.session)
            worker.cancel = cancelled
            out.put({"type": "session", "session": session.id})
            for event in session.chat(req.message, req.max_tokens, cancel=cancelled.is_set):
                if event["type"] == "done":
                    event["governor"] = mack.governor.status()
                    event["memory"] = mack.engine.memory()
                out.put(event)
            mack.save(session)

        out = worker.submit(job)

        def stream():
            try:
                while (event := out.get()) is not _END:
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                # Client went away mid-reply: stop generating instead of finishing unseen.
                cancelled.set()

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/explain")
    def explain(req: ExplainRequest):
        if (r := not_ready()) is not None:
            return r
        cancelled = threading.Event()

        def job(out: queue.Queue):
            worker.cancel = cancelled
            for event in worker.mack.explain(req.text, req.style, req.question, cancel=cancelled.is_set):
                out.put(event)

        out = worker.submit(job)

        def stream():
            try:
                while (event := out.get()) is not _END:
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                cancelled.set()  # panel closed or a new selection replaced this one

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8082) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
