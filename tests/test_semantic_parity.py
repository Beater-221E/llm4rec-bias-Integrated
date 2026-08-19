"""Semantic-parity tests for MiniOneRec reproduction (SFT tokenization + RL reward)."""

from __future__ import annotations

import math

from llm4rec.data.minionerec_rl import (
    build_minionerec_reproduction_rl_eval,
    build_minionerec_reproduction_rl_train,
    rl_dataset_counts,
)
from llm4rec.data.minionerec_prompts import (
    ALPACA_FUSION_INSTRUCTION,
    ALPACA_SFT_INSTRUCTION,
    format_minionerec_alpaca_prompt,
    format_minionerec_rl_prompt,
    fusion_prompt,
    sid2title_prompt,
    sid_sft_prompt,
    title2sid_prompt,
)
from llm4rec.data.minionerec_rl import (
    rl_seq_title2sid_example,
    sid_rl_example,
)
from llm4rec.data.minionerec_sft import (
    MiniOneRecReferenceSFTDataset,
    build_sft_rows,
    encode_reference,
    fusion_seqrec_example,
    sid_item_feat_examples,
    sid_sft_example,
)
from llm4rec.trainers.rollouts import ConstrainedBeamRollout
from llm4rec.trainers.rewards import (
    make_minionerec_reward,
    make_minionerec_reference_reward,
    reference_ndcg_rule_reward,
    reference_rule_reward,
)
from llm4rec.trainers.sft import PadCollator


class FakeTokenizer:
    def __init__(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.apply_chat_template_calls = 0
        self.raw_texts: list[str] = []
        self._vocab: dict[str, int] = {}
        self._next = 10

    def encode(self, text):
        self.raw_texts.append(text)
        ids = []
        for tok in text.split(" "):
            if tok not in self._vocab:
                self._vocab[tok] = self._next
                self._next += 1
            ids.append(self._vocab[tok])
        return ids

    def apply_chat_template(self, *a, **k):
        self.apply_chat_template_calls += 1
        return [1, 2, 3]


class FakeTable:
    levels = 3

    def __contains__(self, item):
        return True

    def sid(self, item):
        return f"<a_{item}><b_{item}><c_{item}>"

    def items(self):
        return ["1", "2"]


# Goldens match ``minionerec_prompts`` (MiniOneRec @ 0c64b955 templates).
UPSTREAM_SID_SFT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n"
    "### Instruction:\n"
    "Can you predict the next possible item that the user may expect?\n"
    "### User Input: \n"
    "The user has interacted with items <a_1><b_2><c_3> in chronological order. "
    "Can you predict the next possible item that the user may expect?\n"
    "### Response:\n"
)
UPSTREAM_TITLE2SID = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n"
    "### Instruction:\n"
    "Answer the question about item identification.\n"
    "### User Input: \n"
    "Which item has the title: Book?\n"
    "### Response:\n"
)
UPSTREAM_SID2TITLE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n"
    "### Instruction:\n"
    "Answer the question about item identification.\n"
    "### User Input: \n"
    'What is the title of item "<a_1><b_2><c_3>"?\n'
    "### Response:\n"
)
UPSTREAM_FUSION = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n"
    "### Instruction:\n"
    "Can you recommend the next item for the user based on their interaction history?\n"
    "### User Input: \n"
    "The user has sequentially interacted with items <a_1><b_2><c_3>. "
    "Can you recommend the next item for him? Tell me the title of the item\n"
    "### Response:\n"
)
UPSTREAM_RL_SID = (
    "### User Input: \n"
    "The user has interacted with items <a_1><b_2><c_3> in chronological order. "
    "Can you predict the next possible item that the user may expect?\n"
    "### Response:\n"
)
UPSTREAM_RL_TITLE = "### User Input: \nWhich item has the title: Book?\n### Response:\n"
UPSTREAM_RL_DESC = (
    "### User Input: \n"
    'An item can be described as follows: "A long story". Which item is it describing?\n'
    "### Response:\n"
)
UPSTREAM_RL_SEQ = (
    "### User Input: \n"
    'Given the title sequence of user historical interactive items: "A", "B", '
    "can you recommend a suitable next item for the user?\n"
    "### Response:\n"
)


def test_minionerec_sft_does_not_use_chat_template():
    tok = FakeTokenizer()
    rows = [{"user_id": "u1", "history": ["1"], "target_item": "2"}]
    exs = build_sft_rows(
        train_rows=rows, sid_table=FakeTable(), meta={}, objectives=["sid_sft"]
    )
    ds = MiniOneRecReferenceSFTDataset(exs, tok, max_len=512)
    _ = ds[0]
    assert tok.apply_chat_template_calls == 0
    assert tok.raw_texts  # raw encode used


def test_minionerec_sid_sft_exact_prompt():
    assert sid_sft_prompt(["<a_1><b_2><c_3>"]) == UPSTREAM_SID_SFT


def test_minionerec_sid_item_feat_exact_prompt():
    assert title2sid_prompt("Book") == UPSTREAM_TITLE2SID
    assert sid2title_prompt("<a_1><b_2><c_3>") == UPSTREAM_SID2TITLE


def test_minionerec_fusion_exact_prompt():
    assert fusion_prompt(["<a_1><b_2><c_3>"]) == UPSTREAM_FUSION
    assert ALPACA_FUSION_INSTRUCTION in UPSTREAM_FUSION


def test_minionerec_rl_prompt_exact():
    ex = sid_rl_example(
        history_sids=["<a_1><b_2><c_3>"], target_sid="<a_9><b_9><c_9>"
    )
    assert ex["prompt"] == UPSTREAM_RL_SID
    assert format_minionerec_rl_prompt("Which item has the title: Book?") == UPSTREAM_RL_TITLE
    assert (
        format_minionerec_rl_prompt(
            'An item can be described as follows: "A long story". Which item is it describing?'
        )
        == UPSTREAM_RL_DESC
    )
    seq = rl_seq_title2sid_example(
        history_titles=["A", "B"], target_sid="<a_9><b_9><c_9>"
    )
    assert seq["prompt"] == UPSTREAM_RL_SEQ


def test_minionerec_upstream_whitespace():
    body = format_minionerec_alpaca_prompt(ALPACA_SFT_INSTRUCTION, "x")
    assert "completes the request.\n### Instruction:" in body
    assert format_minionerec_rl_prompt("hello") == "### User Input: \nhello\n### Response:\n"


def test_minionerec_reference_token_ids():
    tok = FakeTokenizer()
    rows = [{"user_id": "u", "history": ["1"], "target_item": "2"}]
    exs = build_sft_rows(
        train_rows=rows, sid_table=FakeTable(), meta={}, objectives=["sid_sft"]
    )
    ds = MiniOneRecReferenceSFTDataset(exs, tok, max_len=512)
    item = ds[0]
    prompt_ids = encode_reference(tok, exs[0]["prompt_text"], bos=True, eos=False)
    ans_ids = encode_reference(tok, exs[0]["answer_text"], bos=False, eos=True)
    assert item["input_ids"][: len(prompt_ids)] == prompt_ids
    assert item["input_ids"][-len(ans_ids) :] == ans_ids
    assert item["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert item["labels"][-len(ans_ids) :] == ans_ids
    assert sum(item["attention_mask"]) == len(item["input_ids"])


def test_minionerec_left_padding():
    pad = PadCollator(pad_token_id=0, padding_side="left")
    feats = [
        {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]},
        {"input_ids": [3], "labels": [3], "attention_mask": [1]},
    ]
    batch = pad(feats)
    assert batch["input_ids"][1].tolist() == [0, 3]
    assert batch["labels"][1].tolist() == [-100, 3]
    assert batch["attention_mask"][1].tolist() == [0, 1]


def test_minionerec_multiple_windows_same_user_preserved():
    rows = [
        {"user_id": "u1", "history": ["1"], "target_item": "2"},
        {"user_id": "u1", "history": ["1", "2"], "target_item": "1"},
        {"user_id": "u1", "history": ["2"], "target_item": "1"},
    ]
    exs = build_sft_rows(
        train_rows=rows, sid_table=FakeTable(), meta={}, objectives=["sid_sft"]
    )
    assert len(exs) == 3


def test_minionerec_sid_item_feat_cardinality_matches_reference():
    meta = {"1": {"title": "Same"}, "2": {"title": "Same"}}
    exs = build_sft_rows(
        train_rows=[],
        sid_table=FakeTable(),
        meta=meta,
        objectives=["sid_item_feat"],
    )
    # Duplicate titles collapse via dict; unique SIDs 2, unique titles 1
    assert len(exs) == 3


def test_minionerec_validation_uses_sid_sft():
    # Reproduction validation builder = sid_sft objective only
    rows = [{"user_id": "v", "history": ["1"], "target_item": "2"}]
    exs = build_sft_rows(
        train_rows=rows, sid_table=FakeTable(), meta={}, objectives=["sid_sft"]
    )
    assert all(e["objective"] == "sid_sft" for e in exs)


# ---------------------------------------------------------------- RL datasets


def test_minionerec_rl_dataset_mixture():
    rows = [
        {"user_id": "u", "history": ["1"], "target_item": "2"},
    ]
    meta = {"1": {"title": "A", "description": "desc a"}, "2": {"title": "B", "description": "desc b"}}
    out = build_minionerec_reproduction_rl_train(
        train_rows=rows,
        sid_table=FakeTable(),
        meta=meta,
        datasets={"sid_seq": {"sample": -1}, "title_to_sid": {"sample": -1}, "seq_title_to_sid": {"sample": -1}},
        seed=1,
    )
    counts = rl_dataset_counts(out)
    assert counts["sid"] == 1
    assert counts["title2sid"] == 2
    assert counts["description2sid"] == 2
    assert counts["seq_title2sid"] == 1
    for ex in out:
        assert "reference_target_text" in ex
        assert isinstance(ex["prompt"], str)
        assert "### User Input:" in ex["prompt"]
    a = [e["objective"] for e in out]
    b = [
        e["objective"]
        for e in build_minionerec_reproduction_rl_train(
            train_rows=rows,
            sid_table=FakeTable(),
            meta=meta,
            datasets={"sid_seq": {"sample": -1}, "title_to_sid": {"sample": -1}, "seq_title_to_sid": {"sample": -1}},
            seed=2,
        )
    ]
    assert a != b


def test_minionerec_rl_title2sid_prompt():
    meta = {"1": {"title": "Book"}, "2": {"title": "Pen"}}
    out = build_minionerec_reproduction_rl_train(
        train_rows=[], sid_table=FakeTable(), meta=meta,
        datasets={"sid_seq": {"sample": 0}, "title_to_sid": {"sample": -1}, "seq_title_to_sid": {"sample": 0}},
    )
    t = next(e for e in out if e["objective"] == "title2sid")
    assert "Which item has the title:" in t["prompt"]
    d = next(e for e in out if e["objective"] == "description2sid")
    assert "An item can be described as follows:" in d["prompt"]


def test_minionerec_rl_seq_title2sid_prompt():
    rows = [{"user_id": "u", "history": ["1", "2"], "target_item": "1"}]
    meta = {"1": {"title": "A"}, "2": {"title": "B"}}
    out = build_minionerec_reproduction_rl_train(
        train_rows=rows, sid_table=FakeTable(), meta=meta,
        datasets={"sid_seq": {"sample": 0}, "title_to_sid": {"sample": 0}, "seq_title_to_sid": {"sample": -1}},
    )
    s = next(e for e in out if e["objective"] == "seq_title2sid")
    assert 'Given the title sequence of user historical interactive items: "A", "B"' in s["prompt"]


def test_minionerec_rl_target_text_preserved():
    rows = [{"user_id": "u", "history": ["1"], "target_item": "2"}]
    out = build_minionerec_reproduction_rl_train(
        train_rows=rows, sid_table=FakeTable(), meta={},
        datasets={"sid_seq": {"sample": -1}, "title_to_sid": {"sample": 0}, "seq_title_to_sid": {"sample": 0}},
    )
    assert out[0]["reference_target_text"].endswith("\n")


# ---------------------------------------------------------------- RL reward


def test_minionerec_rule_reward_reference_parity():
    comps = ["<a_1>", "bad", "<a_2>\n", '"<a_3>"']
    targets = ["<a_1>", "<a_1>", "<a_2>", "<a_3>"]
    assert reference_rule_reward(comps, targets) == [1.0, 0.0, 1.0, 1.0]


def test_minionerec_ranking_no_hit_returns_zero_component():
    comps = ["a", "b", "c", "d"]
    targets = ["x"] * 4
    assert reference_ndcg_rule_reward(comps, targets, 4) == [0.0, 0.0, 0.0, 0.0]


def test_minionerec_ranking_hit_penalizes_nonhits():
    g = 4
    raw = [-1.0 / math.log2(i + 2) for i in range(g)]
    weights = [-x / sum(raw) for x in raw]
    comps = ["x", "a", "b", "c"]
    targets = ["x"] * g
    out = reference_ndcg_rule_reward(comps, targets, g)
    assert out[0] == 0.0
    for j in (1, 2, 3):
        assert abs(out[j] - weights[j]) < 1e-6


def test_minionerec_ranking_multiple_hits():
    g = 4
    raw = [-1.0 / math.log2(i + 2) for i in range(g)]
    weights = [-x / sum(raw) for x in raw]
    comps = ["x", "x", "b", "c"]
    targets = ["x"] * g
    out = reference_ndcg_rule_reward(comps, targets, g)
    assert out[0] == 0.0 and out[1] == 0.0
    assert abs(out[2] - weights[2]) < 1e-6
    assert abs(out[3] - weights[3]) < 1e-6


def test_minionerec_reproduction_no_invalid_minus_one_penalty():
    rollout = type(
        "R",
        (),
        {
            "texts": ["garbage-text", "<a_1>"],
            "example": {"reference_target_text": "<a_1>\n"},
        },
    )()
    fn = make_minionerec_reference_reward({}, kind="ranking")
    out = fn(rollout)
    assert -1.0 not in out
    assert out[1] >= 1.0  # rule hit


def test_official_rule_reward_matches_sid_without_eos():
    assert reference_rule_reward(["<a_1><b_2><c_3>"], ["<a_1><b_2><c_3>\n"]) == [1.0]


def test_official_rule_reward_misses_if_im_end_kept():
    assert reference_rule_reward(
        ["<a_1><b_2><c_3><|im_end|>"], ["<a_1><b_2><c_3>\n"]
    ) == [0.0]


def test_minionerec_reward_defaults_to_official_ranking():
    rollout = type(
        "R",
        (),
        {
            "texts": ["<a_1>", "miss"],
            "example": {"reference_target_text": "<a_1>\n"},
        },
    )()
    fn = make_minionerec_reward(None, {"reward": {"type": "ranking"}})
    out = fn(rollout)
    assert out[0] >= 1.0
    assert -1.0 not in out


def test_minionerec_reproduction_sft_objectives():
    table = FakeTable()
    meta = {"1": {"title": "Book A"}, "2": {"title": "Book B"}}
    rows = [{"user_id": "u1", "history": ["1"], "target_item": "2"}]
    exs = build_sft_rows(
        train_rows=rows,
        meta=meta,
        sid_table=table,
        objectives=["sid_sft", "sid_item_feat", "fusion_seqrec"],
    )
    tasks = {e["objective"] for e in exs}
    assert {"sid_sft", "fusion_seqrec", "sid_item_feat"} <= tasks
    sid = sid_sft_example(user_id="u", history=["1"], target="2", sid_table=table)
    assert "chronological order" in sid["prompt"][0]["content"]
    assert sid["answer"].startswith("<a_")
    fusion = fusion_seqrec_example(
        user_id="u", history=["1"], target="2", sid_table=table, meta=meta
    )
    assert "Tell me the title" in fusion["prompt"][0]["content"]
    assert "Book B" in fusion["answer"]
    assert len(sid_item_feat_examples(item="1", sid_table=table, meta=meta)) == 2


def test_constrained_rollout_defaults_do_sample_true():
    r = ConstrainedBeamRollout(FakeTable())
    assert r.do_sample is True
    assert r.temperature == 1.0
    assert ConstrainedBeamRollout(FakeTable(), do_sample=False).do_sample is False
