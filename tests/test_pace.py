"""Phase 4: pacing, battery and the slowdown contract. Guarantee 2 (polite). Uses a fake clock."""

import json

from mack import system
from mack.governor import Governor
from mack.tasks import PACES

from .batch_helpers import FakeClock, create, make_env, quiet_mac, run, write_notes  # noqa: F401

REPLY = json.dumps({"text": " ".join(["w"] * 99)})  # about 100 generated tokens


def timed_job(tmp_path, pace, n=100):
    clock = FakeClock()
    cfg, store, bstore, engine = make_env(tmp_path / pace, responder=lambda p: REPLY, clock=clock)
    write_notes(tmp_path / pace / "vault", n, text=lambda i: f"Note {i}. " + "filler words " * 110)
    task, _ = create(cfg, store, bstore, "Summarize.", [str(tmp_path / pace / "vault")], pace=pace)
    start = clock()
    runner = run(cfg, store, bstore, engine, task, clock=clock)
    elapsed = clock() - start
    rows = bstore.items(task.id)
    return cfg, runner, rows, elapsed, engine


def test_gentle_respects_the_cap_including_prefill(tmp_path):
    from mack.power import DEFAULT_ENERGY

    cfg, runner, rows, elapsed, engine = timed_job(tmp_path, "gentle")
    rate = PACES["gentle"][1]  # uncalibrated fallback rate in a fresh state dir
    assert runner.base_rate == rate
    watts = cfg.pace.gentle_watts
    e_pre, e_dec = DEFAULT_ENERGY["prefill_j_per_token"], DEFAULT_ENERGY["decode_j_per_token"]
    # Prompt reading spends the power budget; each generated token waits for the rate cap
    # or the power budget, whichever is slower.
    implied = sum(max(r["tokens_in"] / engine.prefill_tps, e_pre * r["tokens_in"] / watts)
                  + r["tokens_out"] * max(1 / rate, e_dec / watts) for r in rows)
    assert all(r["status"] == "done" for r in rows)
    assert 280 <= rows[0]["tokens_in"] <= 420 and 90 <= rows[0]["tokens_out"] <= 110
    assert elapsed >= 0.97 * implied, (elapsed, implied)
    assert elapsed <= 1.5 * implied  # and it does not dawdle beyond the budget either
    joules = sum(e_pre * r["tokens_in"] + e_dec * r["tokens_out"] for r in rows)
    assert joules / elapsed <= watts * 1.02  # average extra power stays within gentle's budget


def test_full_runs_at_the_engine_speed(tmp_path):
    cfg, runner, rows, elapsed, engine = timed_job(tmp_path, "full")
    unthrottled = sum(r["tokens_in"] / engine.prefill_tps + r["tokens_out"] / engine.tps for r in rows)
    assert runner.base_rate is None
    assert elapsed <= 1.05 * unthrottled + 1.0


def test_job_cap_is_set_and_restored(tmp_path):
    cfg, store, bstore, engine = make_env(tmp_path)
    write_notes(tmp_path / "vault", 3)
    gov = Governor(cfg)
    gov.job_cap = 7.5  # whatever was there before
    chat_cap = gov.rate_cap
    seen = []

    def watch(prompt):
        seen.append(gov.job_cap)
        return json.dumps({"text": "ok"})

    engine.responder = watch
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="balanced")
    run(cfg, store, bstore, engine, task, clock=FakeClock(), governor=gov)
    assert seen and all(c == PACES["balanced"][1] for c in seen)
    assert gov.job_cap == 7.5 and gov.rate_cap == chat_cap

    # pausing mid-job restores it too
    write_notes(tmp_path / "vault2", 3)
    task2, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault2")], pace="gentle")

    def pause_after_first(prompt):
        store.set_control(task2.id, "pause")
        return json.dumps({"text": "ok"})

    engine.responder = pause_after_first
    run(cfg, store, bstore, engine, task2, clock=FakeClock(), governor=gov)
    assert store.load(task2.id).status == "paused" and gov.job_cap == 7.5


def test_low_battery_pauses_with_item_pending_and_recovery_resumes(tmp_path, monkeypatch):
    clock = FakeClock()
    cfg, store, bstore, engine = make_env(tmp_path, clock=clock)
    write_notes(tmp_path / "vault", 40)
    level = {"pct": 80}
    monkeypatch.setattr(system, "power", lambda: system.Power(True, level["pct"]))
    calls = {"n": 0}

    def drain(prompt):
        calls["n"] += 1
        if calls["n"] == 3:
            level["pct"] = 24  # the battery runs down while item 2 is being written
        return REPLY

    engine.responder = drain
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="balanced")
    run(cfg, store, bstore, engine, task, clock=clock)
    t = store.load(task.id)
    assert t.status == "paused" and store.control(task.id) == "battery"
    assert not [r for r in bstore.items(task.id) if r["status"] == "running"]
    paused_at = next(r for r in bstore.items(task.id) if r["status"] == "pending")
    assert paused_at["attempts"] == 0  # a battery pause is not a failed attempt

    level["pct"] = 27
    assert store.next_runnable(cfg) is None  # below the resume level: still waiting
    level["pct"] = 30
    assert store.next_runnable(cfg).id == task.id
    assert store.load(task.id).status == "queued"


def test_on_battery_full_is_demoted_to_steady(tmp_path, monkeypatch):
    clock = FakeClock()
    cfg, store, bstore, engine = make_env(tmp_path, clock=clock)
    write_notes(tmp_path / "vault", 2)
    monkeypatch.setattr(system, "power", lambda: system.Power(True, 90))
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="full")
    runner = run(cfg, store, bstore, engine, task, clock=clock)
    assert runner.rate == PACES["steady"][1]
    assert "on battery" in store.load(task.id).batch["pace_note"]


def test_idle_boost_runs_at_steady_only_while_idle(tmp_path, monkeypatch):
    clock = FakeClock()
    cfg, store, bstore, engine = make_env(tmp_path, clock=clock)
    write_notes(tmp_path / "vault", 2)
    idle = {"s": 900.0}
    monkeypatch.setattr(system, "input_idle_seconds", lambda: idle["s"])
    rates = []

    def watch(prompt):
        rates.append(runner_box[0].current_rate())
        idle["s"] = 0.0  # the user comes back
        clock.advance(2.0)
        return json.dumps({"text": "ok"})

    from mack.batch import BatchRunner

    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="gentle", boost_when_idle=True)
    runner_box = [BatchRunner(engine, cfg, store, store.load(task.id), bstore, clock=clock)]
    engine.responder = watch
    for w in runner_box[0].steps():
        clock.advance(w)
    assert rates[0] == PACES["steady"][1] and rates[1] == PACES["gentle"][1]


def test_thermal_pressure_rests(tmp_path, monkeypatch):
    clock = FakeClock()
    cfg, store, bstore, engine = make_env(tmp_path, clock=clock)
    write_notes(tmp_path / "vault", 1)
    states = iter([system.THERMAL_SERIOUS, system.THERMAL_SERIOUS, system.THERMAL_NOMINAL])
    monkeypatch.setattr(system, "thermal_state", lambda: next(states, system.THERMAL_NOMINAL))
    task, _ = create(cfg, store, bstore, "x", [str(tmp_path / "vault")], pace="full")
    start = clock()
    run(cfg, store, bstore, engine, task, clock=clock)
    assert clock() - start >= 120 and store.load(task.id).status == "done"
