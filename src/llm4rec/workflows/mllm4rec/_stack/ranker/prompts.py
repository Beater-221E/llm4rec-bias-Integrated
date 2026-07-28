# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
# (dataloader/utils.py Prompter + seq_to_token_ids formatting)
#
# Original behavior is preserved unless explicitly documented.

"""Alpaca-short prompting for multimodal item ranking."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


class Prompter:
    def __init__(self, template_name: str = "alpaca_short") -> None:
        pkg = "llm4rec.workflows.mllm4rec._stack.templates"
        try:
            text = resources.files(pkg).joinpath(f"{template_name}.json").read_text(
                encoding="utf-8"
            )
            self.template = json.loads(text)
        except Exception:
            # fallback to file next to package
            path = Path(__file__).resolve().parents[1] / "templates" / f"{template_name}.json"
            self.template = json.loads(path.read_text(encoding="utf-8"))

    def generate_prompt(
        self,
        instruction: str,
        input_text: str | None = None,
        label: str | None = None,
    ) -> str:
        if input_text:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input_text
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        return res


DEFAULT_ML100K_SYSTEM = (
    "You are going to play a movie enthusiast who is selecting movies they are interested in. "
    "I will provide you with the movies you have previously browsed in chronological order,"
    "candidates pool, and vision descriptions for each item in thepreviously browsed item and "
    "candidates pool. The provided data is in the format {item : vision descriptions}. please "
    "correlate the item with the corresponding vision descriptions. Recommend an item from the "
    "candidate pool  with its index letter."
)

DEFAULT_INPUT_TEMPLATE = "Previously browsed: {}; \n Candidate pool: {}"


def truncate_title(tokenizer, title: str, max_title_len: int) -> str:
    tokens = tokenizer.tokenize(str(title))[:max_title_len]
    return tokenizer.convert_tokens_to_string(tokens)


def format_seq_and_candidates(
    *,
    seq: list[int],
    candidates: list[int],
    label: int,
    text_dict: dict[Any, str],
    text_img_dict: dict[Any, str],
    tokenizer,
    max_title_len: int,
    system_template: str,
    input_template: str,
) -> dict[str, str]:
    seq_t = " \n ".join(
        [
            "("
            + str(idx + 1)
            + ") {"
            + truncate_title(tokenizer, text_dict[item], max_title_len)
            + " : "
            + truncate_title(tokenizer, text_img_dict[item], max_title_len)
            + "}"
            for idx, item in enumerate(seq)
        ]
    )
    can_t = " \n ".join(
        [
            "("
            + chr(ord("A") + idx)
            + ") {"
            + truncate_title(tokenizer, text_dict[item], max_title_len)
            + " : "
            + truncate_title(tokenizer, text_img_dict[item], max_title_len)
            + "}"
            for idx, item in enumerate(candidates)
        ]
    )
    output = chr(ord("A") + candidates.index(label))
    return {
        "system": system_template,
        "input": input_template.format(seq_t, can_t),
        "output": output,
    }
