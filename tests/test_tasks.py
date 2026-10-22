import json

import pytest

from mack.config import Config
from mack.tasks import (
    TaskRunner,
    TaskScheduler,
    TaskStore,
    code_interfaces,
    extract_code,
    guess_kind,
    missing_names,
    parse_code_plan,
    parse_report_plan,
    repeating_tail,
)

from .fakes import FakeEngine

REPORT_PLAN = """TITLE: Urban Heat Islands
SECTION: Introduction | 400 words | Why cities run hotter than their surroundings.
- SECTION: Causes | 600 words | Surfaces, waste heat and lost vegetation.
SECTION: Conclusion | 300 | What cities can do."""

SEC1 = "## Introduction\n\nCities in summer run up to 7 degrees warmer than nearby farmland. Asphalt stores heat."
SEC2 = "## Causes\n\nDark roofs absorb sunlight. Air conditioners release waste heat into streets."
SEC3 = "## Conclusion\n\nCool roofs and trees reduce the effect."


def make(tmp_path, replies, prompt="Write a report on urban heat islands", kind="report", pages=3):
    cfg = Config()
    cfg.paths.state = str(tmp_path / "state")
    store = TaskStore(cfg)
    store.output_root = tmp_path / "output"
    task = store.create(prompt, kind=kind, pages=pages, pace="full")
    engine = FakeEngine(replies=replies)
    return cfg, store, task, engine


def run_all(runner):
    waits = list(runner.steps())
    return waits


# ---- parsing -------------------------------------------------------------------


def test_parse_report_plan_is_forgiving():
    title, units = parse_report_plan(REPORT_PLAN)
    assert title == "Urban Heat Islands"
    assert [u.title for u in units] == ["Introduction", "Causes", "Conclusion"]
    assert [u.target_words for u in units] == [400, 600, 300]
    assert units[1].spec.startswith("Surfaces")


def test_parse_code_plan_rejects_unsafe_paths_and_duplicates():
    text = """PROJECT: todo
STACK: Python 3, stdlib
FILE: todo/store.py | Persistence | Store, load_items
FILE: ../evil.py | escape | x
FILE: /etc/passwd | absolute | x
FILE: todo/store.py | duplicate | y
FILE: `README.md` | docs |"""
    project, stack, units = parse_code_plan(text)
    assert project == "todo" and stack.startswith("Python")
    assert [u.title for u in units] == ["todo/store.py", "README.md"]
    assert units[0].interfaces == "Store, load_items"


def test_guess_kind():
    assert guess_kind("Build a Flask app for tracking habits") == "code"
    assert guess_kind("Write a 20 page history of the printing press") == "report"


def test_code_helpers():
    src = 'class Store:\n    """Keeps items."""\n    def __init__(self, path): pass\n    def add(self, item): pass\n\ndef load_items(path, *, strict=False):\n    return []\n'
    sig = code_interfaces("store.py", src)
    assert sig[0].startswith("class Store: __init__(self, path), add(self, item)") and "Keeps items" in sig[0]
    assert sig[1] == "def load_items(path, *, strict=False)"
    assert missing_names("store.py", src, "Store, load_items, save_items") == ["save_items"]
    assert extract_code("Here:\n```python\nx = 1\n```\nDone") == "x = 1\n"
    assert extract_code("```python\nx = 1\ny = 2") == "x = 1\ny = 2\n"  # cut off before the closing fence


def test_repeating_tail():
    good = " ".join(f"sentence {i} is different." for i in range(100))
    assert repeating_tail(good) is None
    # a transcript that repeats itself once is not a loop
    once = good + " We spent most of the time on quarterly planning and the load test. " * 2 + good[:400]
    assert repeating_tail(once) is None
    loop = good + " " + ("The same thing again and again and again. " * 12)
    cut = repeating_tail(loop)
    assert cut is not None and cut < len(loop)


# ---- runs ----------------------------------------------------------------------


def test_report_runs_end_to_end(tmp_path):
    replies = [
        REPORT_PLAN,
        SEC1, "OK", "- Cities run up to 7 degrees warmer than farmland.\n- Cities have 12 moons.",
        SEC2, "- The section repeats the introduction.", SEC2 + " Trees are missing from dense areas.",
        "- Dark roofs absorb sunlight.",
        SEC3, "OK", "- Cool roofs help.",
    ]
    cfg, store, task, engine = make(tmp_path, replies)
    runner = TaskRunner(engine, cfg, store, task)
    run_all(runner)
    t = store.load(task.id)
    assert t.status == "done", t.error
    assert [u.status for u in t.units] == ["done"] * 3
    report = (tmp_path / "output").glob("*/report.md")
    text = next(report).read_text()
    assert text.startswith("# Urban Heat Islands") and "## Causes" in text and "Trees are missing" in text
    # the summary of a unit is verified against the unit: the invented moons are dropped
    assert t.units[0].digest == ["Cities run up to 7 degrees warmer than farmland."]
    assert any("revised" in n for n in t.units[1].notes)
    assert t.tokens > 0 and t.progress == 1.0
    assert (tmp_path / "output").glob("*/MACK_NOTES.md")


def test_code_task_repairs_syntax_and_never_writes_outside(tmp_path):
    plan = "PROJECT: tally\nSTACK: Python\nFILE: tally/core.py | counting | count_words\nFILE: README.md | docs |"
    broken = "```python\ndef count_words(text)\n    return len(text.split())\n```"
    fixed = "```python\ndef count_words(text):\n    return len(text.split())\n```"
    readme = "```markdown\n# tally\nCounts words.\n```"
    cfg, store, task, engine = make(tmp_path, [plan, broken, fixed, readme], prompt="Build a python cli that counts words", kind="code")
    run_all(TaskRunner(engine, cfg, store, task))
    t = store.load(task.id)
    assert t.status == "done", t.error
    out = next((tmp_path / "output").iterdir())
    assert (out / "tally/core.py").read_text().startswith("def count_words(text):")
    assert t.units[0].digest == ["def count_words(text)"]
    assert any("syntax error" in n for n in t.units[0].notes)
    assert "never run" in (out / "MACK_NOTES.md").read_text()


def test_pause_and_resume_does_not_redo_finished_units(tmp_path):
    replies = [REPORT_PLAN, SEC1, "OK", "- Asphalt stores heat."]
    cfg, store, task, engine = make(tmp_path, replies)
    runner = TaskRunner(engine, cfg, store, task)
    steps = runner.steps()
    # run until the first section is done, then pause
    while not any(u.status == "done" for u in runner.task.units):
        next(steps)
    store.set_control(task.id, "pause")
    for _ in steps:
        pass
    t = store.load(task.id)
    assert t.status == "paused" and [u.status for u in t.units] == ["done", "pending", "pending"]

    store.set_control(task.id, "resume")
    assert store.load(task.id).status == "queued"
    engine.replies = [SEC2, "OK", "- Dark roofs absorb sunlight.", SEC3, "OK", "- Cool roofs help."]
    prompts_before = len(engine.summary_prompts)
    run_all(TaskRunner(engine, cfg, store, store.load(task.id)))
    t = store.load(task.id)
    assert t.status == "done" and not engine.replies  # exactly the remaining work was done
    assert len(engine.summary_prompts) == prompts_before


def test_cancel_and_store_rules(tmp_path):
    cfg, store, task, engine = make(tmp_path, [REPORT_PLAN])
    store.set_control(task.id, "cancel")
    assert store.load(task.id).status == "cancelled"
    assert store.next_runnable() is None
    with pytest.raises(ValueError):
        store.create("   ")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("mine")
    with pytest.raises(ValueError):
        store.create("Write a report", out_dir=str(occupied))
    assert (occupied / "keep.txt").read_text() == "mine"


def test_scheduler_runs_oldest_first_and_paces(tmp_path):
    cfg, store, first, engine = make(tmp_path, [REPORT_PLAN, SEC1, "OK", "- a", SEC2, "OK", "- b", SEC3, "OK", "- c"])
    second = store.create("Second report about rivers", pages=2, pace="full")
    sched = TaskScheduler(engine, cfg, store)
    assert sched.tick() is not None and sched.active()["id"] == first.id
    while store.load(first.id).status != "done":
        sched.tick()
    assert store.load(second.id).status == "queued"

    # a paced runner yields roughly 1/rate between tokens
    from mack import pacing

    runner = TaskRunner(engine, cfg, store, second)
    runner.pacer = pacing.fixed(50.0)  # pin the rate: this laptop may be on battery
    engine.replies = ["TITLE: Rivers\nSECTION: A | 300 | a\nSECTION: B | 300 | b"]
    import time

    waits = []
    for w in runner._generate("s", "u", 40, temp=0.0):
        waits.append(w)
        time.sleep(w)  # the worker honours the wait (or serves a request in the meantime)
    gaps = [w for w in waits if w > 0]
    assert gaps and 0.005 < sum(gaps) / len(gaps) <= 0.021


def test_only_one_worker_lock_at_a_time(tmp_path):
    from mack.tasks import WorkerLock

    cfg = Config()
    cfg.paths.state = str(tmp_path / "state")
    a, b = WorkerLock(cfg), WorkerLock(cfg)
    assert a.acquire() and not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_second_scheduler_does_no_work(tmp_path):
    cfg, store, task, engine = make(tmp_path, [REPORT_PLAN])
    first, second = TaskScheduler(engine, cfg, store), TaskScheduler(engine, cfg, store)
    assert first.tick() is not None
    assert second.tick() is None and second.active() is None
    first.lock.release()
