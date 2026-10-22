"""The auto pace: fast while you're away, gentle while you're busy, paused when it would hurt."""

import json
from mack import system
from mack.engine import LazyEngine
from mack.pacing import Pacer
from mack.tasks import PACES

from .batch_helpers import FakeClock, create, make_env, quiet_mac, run, write_notes  # noqa: F401

REPLY = json.dumps({"text": "ok"})


class FakeGovernor:
    def __init__(self, duty):
        self.duty, self.job_cap = duty, None

    def set_job_cap(self, rate):
        previous, self.job_cap = self.job_cap, rate
        return previous

    def restore_job_cap(self, previous):
        self.job_cap = previous


class Mac:
    """A controllable machine: idle time, other apps' CPU duty, power, heat, memory."""

    def __init__(self, monkeypatch, idle=0.0, duty=0.0, battery=False, thermal=0, memory=1):
        self.idle, self.battery, self.thermal, self.memory = idle, battery, thermal, memory
        self.governor = FakeGovernor(duty)
        monkeypatch.setattr(system, "input_idle_seconds", lambda: self.idle)
        monkeypatch.setattr(system, "power", lambda: system.Power(self.battery, 80))
        monkeypatch.setattr(system, "thermal_state", lambda: self.thermal)
        monkeypatch.setattr(system, "memory_pressure", lambda: self.memory)

    @property
    def duty(self):
        return self.governor.duty

    @duty.setter
    def duty(self, v):
        self.governor.duty = v


def pacer(tmp_path, mac, pace="auto"):
    cfg, *_ = make_env(tmp_path)
    clock = FakeClock()
    return Pacer(cfg, pace, mac.governor, clock), clock


def test_levels(tmp_path, monkeypatch):
    mac = Mac(monkeypatch)
    p, clock = pacer(tmp_path, mac)

    def now():
        clock.advance(1.1)
        return p.current()

    mac.idle, mac.duty = 120, 0.05
    st = now()
    assert (st.level, st.rate, st.chunk) == ("away", None, 512)  # full speed, long chunks
    mac.battery = True
    assert now().rate == PACES["steady"][1]  # away on battery: steady
    mac.battery, mac.idle, mac.duty = False, 0, 0.10
    st = now()
    assert (st.level, st.rate, st.chunk) == ("active", PACES["balanced"][1], 128)
    mac.duty = 0.28  # one busy browser tab: still just "active"
    assert now().level == "active"
    mac.duty = 0.40
    assert (now().level, p.state.rate) == ("busy", PACES["gentle"][1])
    mac.duty = 0.95
    assert (now().level, p.state.rate) == ("hold", 0.0)
    mac.duty, mac.idle = 0.40, 300  # idle but other apps busy (a render, a build): not "away"
    assert now().level == "busy"


def test_heat_and_memory(tmp_path, monkeypatch):
    mac = Mac(monkeypatch, idle=300, duty=0.0, thermal=system.THERMAL_FAIR)
    p, clock = pacer(tmp_path, mac)
    clock.advance(2)
    st = p.current()
    assert st.rate == PACES["balanced"][1] and "warm" in st.note  # fair: capped at balanced
    mac.thermal = system.THERMAL_SERIOUS
    clock.advance(2)
    assert p.current().rate == 0.0
    mac.thermal = system.THERMAL_NOMINAL
    clock.advance(30)
    assert p.current().rate == 0.0  # holds a full minute before checking again
    clock.advance(31)
    assert p.current().rate is None
    mac.memory = system.MEMORY_WARN
    clock.advance(2)
    assert p.current().rate is None  # "warn" is normal on a 16 GB Mac
    mac.memory = system.MEMORY_CRITICAL
    clock.advance(2)
    st = p.current()
    assert st.rate == 0.0 and st.unload


def test_fixed_paces_keep_their_rate_but_still_hold(tmp_path, monkeypatch):
    mac = Mac(monkeypatch, idle=0, duty=0.5)
    p, clock = pacer(tmp_path, mac, pace="steady")
    clock.advance(2)
    assert p.current().rate == PACES["steady"][1]  # busy does not slow a fixed pace
    mac.duty = 0.95
    clock.advance(2)
    assert p.current().rate == 0.0  # but heavy load still pauses it


def test_slows_within_two_seconds_of_you_coming_back(tmp_path, monkeypatch):
    clock = FakeClock()
    mac = Mac(monkeypatch, idle=300, duty=0.0)
    cfg, store, bstore, engine = make_env(tmp_path, clock=clock)
    write_notes(tmp_path / "vault", 1, text=lambda i: "word " * 50)
    seen = []
    reply = json.dumps({"text": " ".join(["w"] * 120)})

    def watch(prompt):
        return reply

    engine.responder = watch
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="auto")
    from mack.batch import BatchRunner

    runner = BatchRunner(engine, cfg, store, store.load(task.id), bstore, governor=mac.governor, clock=clock)
    back_at = None
    for w in runner.steps():
        clock.advance(w)
        seen.append((clock(), runner.rate))
        if back_at is None and clock() > 1001.0:
            mac.idle, back_at = 0.0, clock()  # you touch the keyboard
    slowed = next(t for t, r in seen if t > back_at and r == PACES["balanced"][1])
    assert slowed - back_at <= 2.0
    assert seen[0][1] is None  # it started at full speed while you were away


def test_heavy_load_pauses_then_resumes(tmp_path, monkeypatch):
    clock = FakeClock()
    mac = Mac(monkeypatch, idle=0, duty=0.95)
    cfg, store, bstore, engine = make_env(tmp_path, responder=lambda p: REPLY, clock=clock)
    write_notes(tmp_path / "vault", 2)
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="auto")
    from mack.batch import BatchRunner

    runner = BatchRunner(engine, cfg, store, store.load(task.id), bstore, governor=mac.governor, clock=clock)
    start = clock()
    for w in runner.steps():
        clock.advance(w)
        if clock() - start > 45:
            mac.duty = 0.1  # the heavy job finishes
    assert clock() - start >= 45 and not engine.calls[:0]
    assert store.load(task.id).status == "done" and bstore.counts(task.id)["done"] == 2
    assert "heavy load" in " ".join(store.load(task.id).log)


class UnloadableStub:
    def __init__(self, engine):
        self._e, self.unloads = engine, 0

    def __getattr__(self, name):
        return getattr(self._e, name)

    def unload(self):
        self.unloads += 1


def test_critical_memory_frees_the_model_and_redoes_the_item(tmp_path, monkeypatch):
    clock = FakeClock()
    mac = Mac(monkeypatch, idle=0, duty=0.0)
    cfg, store, bstore, stub = make_env(tmp_path, clock=clock)
    engine = UnloadableStub(stub)
    write_notes(tmp_path / "vault", 3)
    calls = {"n": 0}

    def squeeze(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            mac.memory = system.MEMORY_CRITICAL  # memory runs out while item 1 is being written
        return json.dumps({"text": " ".join(["w"] * 40)})

    stub.responder = squeeze
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="auto")
    from mack.batch import BatchRunner

    runner = BatchRunner(engine, cfg, store, store.load(task.id), bstore, governor=mac.governor, clock=clock)
    start = clock()
    for w in runner.steps():
        clock.advance(w)
        if clock() - start > 30 and mac.memory == system.MEMORY_CRITICAL:
            mac.memory = system.MEMORY_NORMAL
    assert engine.unloads == 1
    assert store.load(task.id).status == "done"
    assert [r["attempts"] for r in bstore.items(task.id)] == [1, 1, 1]  # the interrupted item was not charged
    assert calls["n"] == 4  # item 1 was started twice


def test_short_prompt_chunks_while_you_are_present(tmp_path, monkeypatch):
    mac = Mac(monkeypatch, idle=0, duty=0.0)
    cfg, store, bstore, engine = make_env(tmp_path, responder=lambda p: REPLY, clock=FakeClock())
    sizes = []
    real = engine.prefill
    engine.prefill = lambda ctx, toks, progress=None: (sizes.append(len(toks)), real(ctx, toks))
    write_notes(tmp_path / "vault", 1, text=lambda i: "lorem ipsum " * 400)
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="auto")
    run(cfg, store, bstore, engine, task, clock=engine.clock, governor=mac.governor)
    assert max(sizes) == 128
    sizes.clear()
    mac.idle = 600
    write_notes(tmp_path / "vault2", 1, text=lambda i: "lorem ipsum " * 400)
    task2, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault2")], pace="auto")
    run(cfg, store, bstore, engine, task2, clock=engine.clock, governor=mac.governor)
    assert max(sizes) == 512


def test_lazy_engine_loads_on_use_and_unloads(monkeypatch, tmp_path):
    monkeypatch.setenv("MACK_STUB_ENGINE", "1")
    cfg, *_ = make_env(tmp_path)
    e = LazyEngine(cfg)
    assert not e.loaded and e.loads == 0
    assert e.count("two words") == 2 and e.loaded and e.loads == 1
    e.unload()
    assert not e.loaded
    e.encode("again")
    assert e.loads == 2


def test_plan_line_gives_a_range_for_auto(tmp_path):
    cfg, store, bstore, engine = make_env(tmp_path)
    write_notes(tmp_path / "vault", 10)
    _, plan = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="auto", dry_run=True)
    assert "if you leave the Mac alone" in plan["text"] and plan["seconds_busy"] > plan["seconds"]


def test_busy_has_hysteresis_and_levels_have_power_budgets(tmp_path, monkeypatch):
    mac = Mac(monkeypatch, idle=0, duty=0.40)
    p, clock = pacer(tmp_path, mac)

    def now():
        clock.advance(1.1)
        return p.current()

    st = now()
    assert st.level == "busy" and st.watts == p.cfg.background.busy_watts
    mac.duty = 0.30  # a dip below the threshold does not flip back immediately
    assert now().level == "busy"
    mac.duty = 0.20
    st = now()
    assert st.level == "active" and st.watts == p.cfg.background.active_watts
    mac.idle, mac.duty = 300, 0.05
    assert now().watts is None  # away on power: uncapped
    mac.battery = True
    assert now().watts == p.cfg.background.away_battery_watts


def test_wordy_free_text_is_trimmed_not_failed():
    from mack.schema import parse_schema, validate

    schema = parse_schema({"type": "object", "properties": {"summary": {"type": "string", "maxLength": 40},
                                                            "vendor": {"type": "string", "maxLength": 10, "x-grounded": True}}})
    out = validate(schema, {"summary": "This note talks about many different things at great length.", "vendor": "Acme"})
    assert len(out["summary"]) <= 40 and out["summary"].endswith("…") and out["_trimmed"] == ["summary"]
    import pytest

    with pytest.raises(ValueError):
        validate(schema, {"summary": "ok", "vendor": "A very long vendor name"})  # grounded values are never cut


def test_memory_release_between_items_does_not_crash_and_backs_off(tmp_path, monkeypatch):
    clock = FakeClock()
    mac = Mac(monkeypatch, idle=0, duty=0.0, memory=system.MEMORY_CRITICAL)  # critical before the first item
    cfg, store, bstore, stub = make_env(tmp_path, responder=lambda p: REPLY, clock=clock)
    engine = UnloadableStub(stub)
    write_notes(tmp_path / "vault", 2)
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="auto")
    from mack.batch import BatchRunner

    runner = BatchRunner(engine, cfg, store, store.load(task.id), bstore, governor=mac.governor, clock=clock)
    start = clock()
    for w in runner.steps():
        clock.advance(w)
        if clock() - start > 10:
            mac.memory = system.MEMORY_WARN
    t = store.load(task.id)
    assert t.status == "done", t.error
    assert engine.unloads == 1 and clock() - start >= 60  # waited at least a minute before reloading
