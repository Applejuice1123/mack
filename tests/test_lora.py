from mack.lora import dataset
from mack.lora.evals import decide, score, summarize

from .fakes import FMT


def test_dataset_is_deterministic_and_disjoint():
    train, valid, evals = dataset.generate(seed=0)
    assert dataset.generate(seed=0)[0] == train
    assert len(train) + len(valid) == sum(dataset.TRAIN_MIX.values())
    assert len(evals) == sum(dataset.EVAL_MIX.values())
    # eval fact types never appear in training
    eval_things = {spec[0].split(" {s}")[0] for spec in dataset.EVAL_FACTS}
    for case in train + valid:
        assert not any(thing in case["user"] for thing in eval_things)


def test_cases_are_consistent_with_what_is_visible():
    train, valid, evals = dataset.generate(seed=3)
    for case in train + valid + evals:
        prompt = dataset.render_prompt(FMT, case)
        assert prompt.endswith("<|im_start|>assistant\n")
        if case["kind"] in ("pinned", "recalled", "visible"):
            # the answer is somewhere the model can see, and the target reply states it
            assert case["gold"] in prompt and case["gold"] in case["reply"]
        if case["kind"] in ("missing", "fresh_missing"):
            assert case["answerable"] is False
            assert score(case, case["reply"]) == "abstained"
        if case["kind"] == "missing_near":
            assert score(case, case["reply"]) == "abstained"
        if case["kind"] == "pinned":
            assert "Exact quotes:" in case["system"]


def test_score_outcomes():
    unanswerable = {"kind": "missing", "answerable": False, "gold": "4821"}
    assert score(unanswerable, "The gate code is 9310.") == "fabricated"
    assert score(unanswerable, "I can't see that part of our conversation anymore.") == "abstained"
    answerable = {"kind": "pinned", "answerable": True, "gold": "$1,250"}
    assert score(answerable, "The budget is $1250.") == "correct"
    assert score(answerable, "I don't have that anymore, could you tell me again?") == "refused"
    assert score(answerable, "It was $900.") == "wrong"
    assert score({"kind": "identity", "answerable": True, "gold": "Mack"}, "I'm Mack, a local assistant.") == "correct"
    assert score({"kind": "identity", "answerable": True, "gold": "Mack"}, "I am Qwen, made by Alibaba.") == "wrong"
    assert score({"kind": "identity", "answerable": True, "gold": "Mac"}, "I run entirely on your Mac.") == "correct"
    assert score({"kind": "general", "answerable": True, "gold": ["6", "six"]}, "A hexagon has six sides.") == "correct"
    assert score({"kind": "general", "answerable": True, "gold": "H2O"}, "The symbol is H₂O.") == "correct"


def rows(fabricated, refused, n=20):
    out = [{"kind": "missing", "outcome": "fabricated" if i < fabricated else "abstained"} for i in range(n)]
    out += [{"kind": "pinned", "outcome": "refused" if i < refused else "correct"} for i in range(n)]
    out += [{"kind": "identity", "outcome": "correct"}, {"kind": "general", "outcome": "correct"}]
    return out


def test_gate_requires_fewer_fabrications_without_more_refusals():
    base = summarize(rows(fabricated=12, refused=0))
    assert decide(base, summarize(rows(fabricated=2, refused=0)))["ship"] is True
    assert decide(base, summarize(rows(fabricated=12, refused=0)))["ship"] is False  # no improvement
    over_refusing = decide(base, summarize(rows(fabricated=0, refused=6)))
    assert over_refusing["ship"] is False and "refuses" in over_refusing["reasons"][0]
