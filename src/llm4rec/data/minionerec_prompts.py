"""Exact MiniOneRec prompt templates (pinned commit ``0c64b955``).

Upstream sources (``AkaliKong/MiniOneRec`` ``data.py``):
  - ``BaseDataset.generate_prompt`` / ``SidSFTDataset.pre`` instruction block
  - ``SidItemFeatDataset.generate_prompt`` / ``pre``
  - ``FusionSeqRecDataset.generate_formatted_prompt`` / ``pre``
  - RL ``SidDataset`` / ``RLTitle2SidDataset`` / ``RLSeqTitle2SidDataset`` wraps

Whitespace is intentional and must match byte-for-byte (tokenizer-sensitive).
In particular, upstream places a trailing space after ``request.`` before ``\\n\\n``.
"""

from __future__ import annotations

from typing import Sequence

# Upstream instruction bodies (without the shared Alpaca preamble).
ALPACA_SFT_INSTRUCTION = "Can you predict the next possible item that the user may expect?"
ALPACA_ITEMFEAT_INSTRUCTION = "Answer the question about item identification."
ALPACA_FUSION_INSTRUCTION = (
    "Can you recommend the next item for the user based on their interaction history?"
)

# Exact shared preamble from SidSFTDataset / SidItemFeat / FusionSeqRec ``pre()``.
# NOTE: trailing space after "request." is required for upstream parity.
_ALPACA_PREAMBLE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n"
)


def format_minionerec_alpaca_prompt(instruction: str, user_input: str) -> str:
    """Full SFT Alpaca string: preamble + Instruction + User Input + Response header."""
    return (
        f"{_ALPACA_PREAMBLE}"
        f"### Instruction:\n{instruction}\n"
        f"### User Input: \n{user_input}\n"
        f"### Response:\n"
    )


def format_minionerec_rl_prompt(user_input: str) -> str:
    """RL truncated wrap matching ``BaseDataset.generate_prompt`` with empty output."""
    return f"### User Input: \n{user_input}\n### Response:\n"


def sid_sft_user_input(history_sids: Sequence[str]) -> str:
    joined = ", ".join(history_sids)
    return (
        f"The user has interacted with items {joined} in chronological order. "
        "Can you predict the next possible item that the user may expect?"
    )


def fusion_user_input(history_sids: Sequence[str]) -> str:
    joined = ", ".join(history_sids)
    return (
        f"The user has sequentially interacted with items {joined}. "
        "Can you recommend the next item for him? Tell me the title of the item"
    )


def title2sid_user_input(title: str) -> str:
    return f"Which item has the title: {title}?"


def sid2title_user_input(sid: str) -> str:
    return f'What is the title of item "{sid}"?'


def description2sid_user_input(description: str) -> str:
    return f'An item can be described as follows: "{description}". Which item is it describing?'


def seq_title2sid_user_input(inter_titles: str) -> str:
    return (
        f"Given the title sequence of user historical interactive items: "
        f"{inter_titles}, can you recommend a suitable next item for the user?"
    )


def sid_sft_prompt(history_sids: Sequence[str]) -> str:
    return format_minionerec_alpaca_prompt(
        ALPACA_SFT_INSTRUCTION, sid_sft_user_input(history_sids)
    )


def fusion_prompt(history_sids: Sequence[str]) -> str:
    return format_minionerec_alpaca_prompt(
        ALPACA_FUSION_INSTRUCTION, fusion_user_input(history_sids)
    )


def title2sid_prompt(title: str) -> str:
    return format_minionerec_alpaca_prompt(
        ALPACA_ITEMFEAT_INSTRUCTION, title2sid_user_input(title)
    )


def sid2title_prompt(sid: str) -> str:
    return format_minionerec_alpaca_prompt(
        ALPACA_ITEMFEAT_INSTRUCTION, sid2title_user_input(sid)
    )
