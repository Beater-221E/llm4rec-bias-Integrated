"""带 SID 词表扩展的模型加载 —— 全参微调，无 LoRA。

三种加载场景：

  ``from_backbone``    从 HF backbone 起步：加 SID token → resize embedding → 初始化新行
  ``from_checkpoint``  从我们自己存的全参 checkpoint 起步（RL 接 SFT、或单独评测）

SID token 加成**普通 token**（``special_tokens=False``），因为 TRL 的 GRPO
在 decode 时会 ``skip_special_tokens=True``，加成 special 会被整段抹掉。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.sid.table import SidTable

_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class SidModelBundle:
    tokenizer: Any
    model: PreTrainedModel
    new_token_ids: list[int]
    n_new_tokens: int


def resolve_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in ("auto", ""):
        return torch.float32
    if key not in _DTYPES:
        raise ConfigurationError(f"未知精度 '{name}'，可用：{sorted(_DTYPES)}")
    return _DTYPES[key]


def resolve_checkpoint(model_cfg: dict[str, Any]) -> str:
    """本地权重优先，其次 HF id。离线机器上只要放好 local_path 就能跑。"""
    local = model_cfg.get("local_path")
    if local:
        path = Path(local)
        if path.is_dir() and any(path.glob("*.safetensors") or path.glob("*.bin")):
            return str(path)
    ckpt = model_cfg.get("checkpoint")
    if not ckpt:
        raise ConfigurationError("model.checkpoint 必填")
    return str(ckpt)


def load_for_sid(
    model_cfg: dict[str, Any],
    table: SidTable,
    *,
    checkpoint_override: str | None = None,
    local_rank: int = 0,
) -> SidModelBundle:
    """加载 backbone 或已有 checkpoint，并确保 SID token 就位。

    ★ 这里【没有】任何 LoRA / PEFT 路径。官方 MiniOneRec 的 SFT 就是全参微调
      （``sft.py`` 里只有一个 ``freeze_LLM`` 开关，没有 peft），而且 LoRA 会
      低估 RL 对表征的改动 —— 我们后续要做表征分析，必须全参。
    """
    checkpoint = checkpoint_override or resolve_checkpoint(model_cfg)
    dtype = resolve_dtype(str(model_cfg.get("dtype") or "fp32"))

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=bool(model_cfg.get("trust_remote_code", False))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sid_tokens = table.all_tokens()
    n_added = tokenizer.add_tokens(sid_tokens, special_tokens=False)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    old_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > old_size:
        model.resize_token_embeddings(len(tokenizer))

    new_ids = [tokenizer.convert_tokens_to_ids(t) for t in sid_tokens]
    if any(i is None or i < 0 for i in new_ids):
        raise ConfigurationError("SID token 加入词表失败")

    if n_added > 0:
        # 新增行用旧词表的均值初始化 + 小噪声。全零初始化会让新 token 的
        # logit 全部相同，前几百步基本学不动。
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean = emb[:old_size].mean(dim=0)
            for idx in new_ids:
                emb[idx] = mean + 0.02 * torch.randn_like(mean)
            out = model.get_output_embeddings()
            if out is not None and out.weight is not emb:
                out_mean = out.weight[:old_size].mean(dim=0)
                for idx in new_ids:
                    out.weight[idx] = out_mean + 0.02 * torch.randn_like(out_mean)

    if bool(model_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # 官方提供的选项：冻结 LLM，只训新增的 SID embedding
    if bool(model_cfg.get("freeze_llm", False)):
        if n_added == 0:
            raise ConfigurationError(
                "freeze_llm=true 但没有新增 token —— 那样所有参数都会被冻住"
            )
        _freeze_except_embeddings(model)

    if torch.cuda.is_available():
        model = model.to(f"cuda:{local_rank}")

    return SidModelBundle(
        tokenizer=tokenizer,
        model=model,
        new_token_ids=new_ids,
        n_new_tokens=int(n_added),
    )


def _freeze_except_embeddings(model: PreTrainedModel) -> None:
    for param in model.parameters():
        param.requires_grad = False
    model.get_input_embeddings().weight.requires_grad = True
    out = model.get_output_embeddings()
    if out is not None:
        out.weight.requires_grad = True


def load_trained(
    checkpoint_dir: str | Path,
    *,
    dtype: str = "fp32",
    local_rank: int = 0,
) -> SidModelBundle:
    """加载我们自己存的全参 checkpoint（SID token 已经在里面了）。

    RL 接 SFT、以及单独评测走这条路 —— 不再有 ``PeftModel.from_pretrained``
    那套 adapter 合并流程。
    """
    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise MissingArtifactError(f"checkpoint 目录不存在：{path}")

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=resolve_dtype(dtype))
    if torch.cuda.is_available():
        model = model.to(f"cuda:{local_rank}")
    return SidModelBundle(
        tokenizer=tokenizer, model=model, new_token_ids=[], n_new_tokens=0
    )
