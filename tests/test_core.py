from types import SimpleNamespace

import pytest

from mack.archive import Archive, has_backreference, passages
from mack.config import Config, load_config
from mack.governor import rate_for_target
from mack.calibrate import sync_threshold
from mack.pins import extract_pins, select_pins
from mack.session import Session
from mack.textutil import clip_middle, entities, numbers, split_sentences
from mack.verify import check_line, verify_summary

from .fakes import FakeEngine


def turn(idx, role, text):
    return SimpleNamespace(idx=idx, role=role, text=text)


# ---- text helpers ------------------------------------------------------------


def test_numbers_normalize():
    assert numbers("It cost $1,200.50, about 15% of the 3 GB budget.") == {"1200.5", "15", "3"}
    assert numbers("Meet on 2026-10-12 at 3pm") == {"2026", "10", "12", "3"}
    assert numbers("no digits here") == set()


def test_entities_skip_sentence_initial_words():
    found = entities("Yesterday Priya met Dr Okafor in Lisbon. The plan uses NASA data.")
    assert {"Priya", "Okafor", "Lisbon", "NASA"} <= found
    assert "Yesterday" not in found and "The" not in found


def test_split_sentences_skips_code_and_bullets():
    text = "First point. Second point!\n- a bullet item\n```\ncode = 1\n```\nLast line"
    assert split_sentences(text) == ["First point.", "Second point!", "a bullet item", "Last line"]


def test_clip_middle_keeps_ends():
    text = "A" * 500 + "B" * 500
    clipped = clip_middle(text, 200)
    assert len(clipped) <= 200 and clipped.startswith("A") and clipped.endswith("B")
    assert clip_middle("short", 200) == "short"


# ---- pins --------------------------------------------------------------------


def test_pins_keep_exact_quotes_of_numbers_names_decisions():
    turns = [
        turn(0, "user", "Hi there! My name is Alex and my budget is $4,500."),
        turn(1, "assistant", "Nice to meet you. How can I help?"),
        turn(2, "user", "Let's go with the Lisbon venue instead of Porto."),
        turn(3, "assistant", "Sounds good."),
    ]
    pins = extract_pins(turns)
    quotes = {p.quote for p in pins}
    assert "My name is Alex and my budget is $4,500." in quotes
    assert "Let's go with the Lisbon venue instead of Porto." in quotes
    assert "Sounds good." not in quotes and "How can I help?" not in quotes
    # every pin is a verbatim substring of its turn
    for p in pins:
        assert p.quote in turns[p.turn].text


def test_select_pins_respects_budget_and_order():
    turns = [turn(i, "user", f"I decided item {i} costs {i * 100} dollars.") for i in range(10)]
    pins = extract_pins(turns)
    chosen = select_pins(pins, budget_tokens=60, count=lambda s: len(s.split()))
    assert 0 < len(chosen) < len(pins)
    assert [p.turn for p in chosen] == sorted(p.turn for p in chosen)
    assert sum(len(p.quote.split()) + 8 for p in chosen) <= 60
    # dedup
    assert len(select_pins(pins + pins, 10_000, lambda s: 1)) == len(pins)


# ---- verification ------------------------------------------------------------

SOURCE = (
    "User: I'm planning a trip to Lisbon in March with a budget of 1,200 euros.\n"
    "Mack: Great, flights from Boston are cheapest on Tuesdays.\n"
    "User: Book the hotel near Alfama, my sister Maria is joining."
)


@pytest.mark.parametrize(
    "line,kept",
    [
        ("The user is planning a trip to Lisbon in March.", True),
        ("The user's budget is 1200 euros.", True),
        ("The user's budget is 2,000 euros.", False),  # wrong number
        ("The user is travelling with their brother Carlos.", False),  # invented name
        ("Mack recommended a vegan restaurant downtown.", False),  # unsupported content
        ("The user's sister Maria is joining the trip.", True),
    ],
)
def test_check_line(line, kept):
    assert check_line(line, SOURCE).kept is kept


def test_verify_summary_drops_only_unsupported_lines():
    summary = (
        "Here is the summary:\n"
        "- The user plans a trip to Lisbon in March.\n"
        "- The budget is 5,000 euros.\n"
        "- Flights from Boston are cheapest on Tuesdays."
    )
    verdicts = verify_summary(summary, SOURCE)
    assert [v.kept for v in verdicts] == [False, True, False, True]


# ---- archive -----------------------------------------------------------------


def test_archive_search_only_returns_older_relevant_turns(tmp_path):
    a = Archive(tmp_path / "a.db")
    a.add("s", 0, "user", "The wifi password for the cabin is tangerine-42.")
    a.add("s", 1, "assistant", "Noted, I will remember the cabin wifi password.")
    a.add("s", 2, "user", "Completely unrelated: what is a good pasta recipe?")
    a.add("other", 0, "user", "The wifi password is hunter2.")
    hits = a.search("s", "what was the wifi password for the cabin again?", before_turn=2)
    assert hits and hits[0].turn == 0 and "tangerine-42" in hits[0].text
    assert all(h.turn < 2 for h in hits)
    assert a.search("s", "wifi password cabin", before_turn=0) == []
    assert a.search("s", "tell me about quantum gravity", before_turn=3) == []
    # statements do not trigger recall, only questions and references to earlier
    assert a.search("s", "the cabin wifi password is annoying to type", before_turn=2) == []
    assert has_backreference("what did you say earlier?") and not has_backreference("hello")


def test_passages_split_long_turns():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = passages(text, 300)
    assert len(parts) > 5 and all(len(p) <= 300 for p in parts)


# ---- governor ----------------------------------------------------------------

CELLS = [
    {"tps": 6.0, "slowdown": 0.02},
    {"tps": 12.0, "slowdown": 0.05},
    {"tps": 20.0, "slowdown": 0.11},
    {"tps": 40.0, "slowdown": 0.30},
]


def test_rate_for_target():
    cap, _ = rate_for_target(CELLS, 0.10)
    assert 12.0 < cap < 20.0
    assert rate_for_target(CELLS, 0.50)[0] is None  # full speed fits
    assert rate_for_target(CELLS, 0.01)[0] == 3.0  # below the lowest cell: interpolate toward (0, 0)
    assert rate_for_target(CELLS, 0.01, floor=4.0)[0] == 4.0  # but never under the floor
    assert rate_for_target([], 0.1) == (None, "uncalibrated")
    # noise: a faster cell measuring lower slowdown must not raise the cap past a worse slower cell
    noisy = [{"tps": 10.0, "slowdown": 0.20}, {"tps": 20.0, "slowdown": 0.05}]
    assert rate_for_target(noisy, 0.10)[0] == 5.0


def test_sync_threshold():
    cells = [
        {"gov_duty": 0.0, "tps_ahead1": 44.0, "tps_sync": 40.0},
        {"gov_duty": 0.25, "tps_ahead1": 36.0, "tps_sync": 34.0},
        {"gov_duty": 0.5, "tps_ahead1": 25.0, "tps_sync": 25.0},
    ]
    t, _ = sync_threshold(cells)
    assert 0.25 < t <= 0.5
    always = [{"gov_duty": d, "tps_ahead1": 40.0, "tps_sync": 30.0} for d in (0.0, 0.5)]
    assert sync_threshold(always)[0] > 1.0


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "mack.toml"
    p.write_text("[context]\nbudget_tokens = 4096\n")
    assert load_config(p).context.budget_tokens == 4096
    p.write_text("[context]\nkv_bits = 4\n")
    with pytest.raises(KeyError):
        load_config(p)


# ---- session -----------------------------------------------------------------


def make_session(tmp_path, budget=400, **kw):
    cfg = Config()
    cfg.context.budget_tokens = budget
    cfg.recall.max_tokens = 120
    engine = FakeEngine(**kw)
    return Session(engine, cfg, Archive(tmp_path / "a.db"), system_prompt="You are Mack."), engine


def drain(session, text, max_tokens=50):
    return list(session.chat(text, max_tokens=max_tokens))


def test_turns_append_without_rebuild(tmp_path):
    s, engine = make_session(tmp_path, budget=4000, replies=["Hello Alex.", "It is sunny."])
    drain(s, "Hi, I am Alex.")
    events = drain(s, "How is the weather?")
    assert engine.prefills == 1  # only the initial system prompt
    assert [t.role for t in s.turns] == ["user", "assistant", "user", "assistant"]
    assert events[-1]["type"] == "done" and events[-1]["reply"] == "It is sunny."
    text = engine.decode(s.ctx.tokens)
    assert text.endswith("It is sunny.<|im_end|>")
    assert text.count("<|im_start|>user") == 2


def test_compaction_rebuilds_with_pins_summary_and_recent_turns(tmp_path):
    s, engine = make_session(
        tmp_path,
        budget=400,
        summary="- The user would love hiking suggestions near the coast.\n"
        "- The user's budget is 9,999 euros.\n- The user owns a dog named Rex.\n"
        "- The user is called Alex and has a budget of 1,200 euros for the trip itinerary.",
    )
    drain(s, "My name is Alex and my budget is 1,200 euros for the trip itinerary. Some hiking suggestions near the coast would be lovely.")
    filler = " ".join(["word"] * 60)
    statuses = []
    for i in range(8):
        statuses += [e for e in drain(s, f"Message {i}: {filler}") if e["type"] == "status"]
    done = [e for e in statuses if e["phase"] == "compact_done"]
    assert done, "compaction should have triggered"
    assert done[0]["tokens_after"] < done[0]["tokens_before"]
    assert s.live_from > 0 and len(s.ctx.tokens) < 400
    context = engine.decode(s.ctx.tokens)
    # exact quote survives, verbatim
    assert '"My name is Alex and my budget is 1,200 euros for the trip itinerary."' in context
    # supported bullet kept, fabricated number and invented dog dropped
    assert "The user would love hiking suggestions near the coast." in s.summary
    assert not any("9,999" in b or "Rex" in b for b in s.summary)
    assert done[0]["summary_dropped"] >= 2
    # a bullet that only repeats a pinned quote is not stored twice
    assert not any("1,200" in b for b in s.summary)
    # the most recent exchange is still verbatim, the oldest filler is gone
    assert "Message 7:" in context and "Message 0:" not in context
    # nothing was destroyed
    assert "1,200 euros" in s.archive.turn_text(s.id, 0)


def test_recall_brings_back_archived_original(tmp_path):
    s, engine = make_session(tmp_path, budget=400, summary="- Small talk.")
    drain(s, "The door code for the studio is 7-7-4-1 and nothing else matters.")
    filler = " ".join(["lorem"] * 60)
    for i in range(8):
        drain(s, f"Note {i}: {filler}")
    assert s.live_from > 0
    s.pins = []  # force the detail out of memory so only recall can supply it
    s._rebuild()
    assert "7-7-4-1" not in engine.decode(s.ctx.tokens)
    events = drain(s, "What was the door code for the studio again?")
    assert any(e["type"] == "status" and e["phase"] == "recall" for e in events)
    assert events[-1]["recalled"] == [1]
    context = engine.decode(s.ctx.tokens)
    assert "[Recalled: original text" in context and "7-7-4-1" in context


def test_oversized_message_is_clipped_live_but_archived_whole(tmp_path):
    s, engine = make_session(tmp_path, budget=400)
    huge = " ".join(f"w{i}" for i in range(2000))
    events = drain(s, huge)
    assert any(e["type"] == "status" and e["phase"] == "clipped" for e in events)
    assert len(s.ctx.tokens) < 800
    assert s.turns[0].text == huge


def test_save_and_load_roundtrip(tmp_path):
    s, engine = make_session(tmp_path, budget=400, summary="- The user wants a trip itinerary.")
    drain(s, "My budget is 1,200 euros for the trip itinerary.")
    for i in range(8):
        drain(s, f"Message {i}: " + " ".join(["word"] * 60))
    path = s.save(tmp_path / "sessions")
    loaded = Session.load(path, engine, s.cfg, s.archive)
    assert loaded.live_from == s.live_from and len(loaded.turns) == len(s.turns)
    assert engine.decode(loaded.ctx.tokens) == s._render_context()
    assert loaded.pins == s.pins and loaded.summary == s.summary


def test_interrupted_reply_is_still_recorded(tmp_path):
    s, engine = make_session(tmp_path, budget=4000, replies=["one two three four five six"])
    gen = s.chat("go")
    for event in gen:
        if event["type"] == "token":
            break
    gen.close()
    assert [t.role for t in s.turns] == ["user", "assistant"]
    assert s.turns[1].text.startswith("one")
