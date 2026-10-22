import pytest

from mack.explain import MAX_INPUT_CHARS, build_messages, clean, explain

from .fakes import FakeEngine


def test_clean_tidies_ocr_text():
    raw = "  The mito-\nchondria   is the\r\n\n\n powerhouse \t of the cell.  "
    assert clean(raw) == "The mitochondria is the\npowerhouse of the cell."
    assert clean(" \n \t ") == ""


def test_build_messages_styles_and_question():
    _, user, clipped = build_messages("E = mc^2", "simpler")
    assert "E = mc^2" in user and "simply" in user and not clipped
    _, user, _ = build_messages("E = mc^2", "explain", question="What is c?")
    assert user.endswith("Answer this question about it: What is c?")
    with pytest.raises(ValueError):
        build_messages("x", "shout")


def test_long_selection_is_clipped():
    _, user, clipped = build_messages("word " * 10_000)
    assert clipped and len(user) < MAX_INPUT_CHARS + 400 and "left out" in user


def test_explain_streams_then_reports_and_keeps_no_state():
    engine = FakeEngine(replies=["This is Einstein's mass-energy equivalence."])
    events = list(explain(engine, "E = mc^2"))
    assert "".join(e["text"] for e in events if e["type"] == "token") == "This is Einstein's mass-energy equivalence."
    done = events[-1]
    assert done["type"] == "done" and done["reply"].startswith("This is") and done["input_tokens"] > 0
    # a second call starts from a fresh context
    engine.replies = ["Second."]
    assert list(explain(engine, "other text"))[-1]["reply"] == "Second."


def test_empty_selection_is_an_error_not_a_guess():
    events = list(explain(FakeEngine(), "   \n  "))
    assert events == [{"type": "error", "message": "There was no text in that selection."}]
