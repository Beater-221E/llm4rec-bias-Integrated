"""带 SID 词表扩展的模型加载 —— 全参微调，无 LoRA。

三种加载场景：

  ``from_backbone``    从 HF backbone 起步：加 SID token → resize embedding → 初始化新行
  ``from_checkpoint``  从我们自己存的全参 checkpoint 起步（RL 接 SFT、或单独评测）

SID token 加成**普通 token**（``special_tokens=False``），因为 TRL 的 GRPO
在 decode 时会 ``skip_special_tokens=True``，加成 special 会被整段抹掉。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from llm4rec.core.exceptions import ConfigurationError, MissingArtifactError
from llm4rec.sid.table import SidTable

_LOG = logging.getLogger(__name__)

_TOKENIZER_MARKERS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "spiece.model",
    "tokenizer.model",
)
_STUB_VOCAB_THRESHOLD = 1000

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


def initialize_added_tokens(
    model: PreTrainedModel,
    *,
    old_vocab_size: int,
    new_token_ids: list[int],
    mode: str = "reference",
) -> str:
    """Initialize newly added SID embedding rows.

    ``reference`` — leave ``resize_token_embeddings`` rows untouched (MiniOneRec).
    ``mean_noise`` — mean of prior rows + small Gaussian noise (integrated experiment).
    """
    mode_l = str(mode or "reference").lower().strip()
    if mode_l in {"reference", "resize", "none", "hf"}:
        return "reference"
    if mode_l not in {"mean_noise", "mean+noise", "mean"}:
        raise ConfigurationError(
            f"unsupported sid_token_initialization='{mode}' "
            "(allowed: reference | mean_noise)"
        )
    if not new_token_ids:
        return "mean_noise"
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean = emb[:old_vocab_size].mean(dim=0)
        for idx in new_token_ids:
            emb[idx] = mean + 0.02 * torch.randn_like(mean)
        out = model.get_output_embeddings()
        if out is not None and out.weight is not emb:
            out_mean = out.weight[:old_vocab_size].mean(dim=0)
            for idx in new_token_ids:
                out.weight[idx] = out_mean + 0.02 * torch.randn_like(out_mean)
    return "mean_noise"


def resolve_sid_token_initialization(model_cfg: dict[str, Any], *, mode: str) -> str:
    """Default: reproduction → reference; integrated → mean_noise (or configured)."""
    raw = model_cfg.get("sid_token_initialization")
    if raw is not None and str(raw).strip():
        return str(raw).lower().strip()
    return "reference" if mode == "reproduction" else "mean_noise"


def load_for_sid(
    model_cfg: dict[str, Any],
    table: SidTable,
    *,
    checkpoint_override: str | None = None,
    local_rank: int = 0,
    attn_implementation: str | None = None,
    experiment_mode: str | None = None,
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

    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    attn = attn_implementation or model_cfg.get("attn_implementation")
    if attn:
        load_kwargs["attn_implementation"] = str(attn)
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)
    except Exception:
        # Eager fallback if sdpa/flash unavailable
        load_kwargs.pop("attn_implementation", None)
        load_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)

    old_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > old_size:
        model.resize_token_embeddings(len(tokenizer))

    new_ids = [tokenizer.convert_tokens_to_ids(t) for t in sid_tokens]
    if any(i is None or i < 0 for i in new_ids):
        raise ConfigurationError("SID token 加入词表失败")

    mode = str(experiment_mode or model_cfg.get("_experiment_mode") or "integrated")
    init_mode = resolve_sid_token_initialization(model_cfg, mode=mode)
    if n_added > 0:
        applied = initialize_added_tokens(
            model,
            old_vocab_size=old_size,
            new_token_ids=new_ids,
            mode=init_mode,
        )
    else:
        applied = init_mode if init_mode == "reference" else "reference"
    model_cfg["_sid_token_initialization_effective"] = applied

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


def _restore_chat_template(tokenizer: Any, backbone: str | Path | None) -> None:
    """HF mid-checkpoints often drop chat_template; copy it back from backbone."""
    if getattr(tokenizer, "chat_template", None) or not backbone:
        return
    src = AutoTokenizer.from_pretrained(backbone)
    if getattr(src, "chat_template", None):
        tokenizer.chat_template = src.chat_template


def _has_tokenizer_files(path: Path) -> bool:
    return any((path / name).is_file() for name in _TOKENIZER_MARKERS)


def _load_tokenizer_for_checkpoint(
    path: Path,
    *,
    backbone: str | Path | None = None,
    sid_table: SidTable | None = None,
) -> Any:
    """Load tokenizer from a full save, or rebuild from backbone + SID tokens.

    HF Trainer mid-checkpoints often keep weights only. ``from_pretrained`` on
    that directory yields a stub Qwen tokenizer (vocab size 1); encoding then
    produces empty ``input_ids`` and beam search crashes.
    """
    source = path if _has_tokenizer_files(path) else None
    if source is None and backbone:
        source = backbone
        _LOG.info("checkpoint %s 没有 tokenizer 文件，从 backbone 重建：%s", path, backbone)
    if source is None:
        raise MissingArtifactError(
            f"checkpoint {path} 没有 tokenizer 文件，且未提供 backbone"
        )

    tokenizer = AutoTokenizer.from_pretrained(source)
    if len(tokenizer) < _STUB_VOCAB_THRESHOLD:
        if not backbone or Path(str(backbone)) == Path(str(source)):
            raise MissingArtifactError(
                f"tokenizer vocab={len(tokenizer)}，无法从 {source} 恢复"
            )
        _LOG.warning(
            "checkpoint tokenizer vocab=%d，视为空壳，改从 backbone 加载：%s",
            len(tokenizer),
            backbone,
        )
        tokenizer = AutoTokenizer.from_pretrained(backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _restore_chat_template(tokenizer, backbone)
    if sid_table is not None:
        n_added = tokenizer.add_tokens(sid_table.all_tokens(), special_tokens=False)
        if n_added:
            _LOG.info("已补 %d 个 SID token，tokenizer vocab=%d", n_added, len(tokenizer))
    return tokenizer


def _assert_tokenizer_matches_model(
    tokenizer: Any,
    model: PreTrainedModel,
    *,
    sid_table: SidTable | None = None,
) -> None:
    """Tokenizer must not be larger than the embedding matrix.

    Qwen2.5 keeps unused padding rows (e.g. 151665 tokens vs 151936 embeddings).
    That gap is hardware padding, not missing SID tokens. SID presence is checked
    separately when ``sid_table`` is provided.
    """
    n_tok = len(tokenizer)
    n_emb = int(model.get_input_embeddings().num_embeddings)
    if n_tok > n_emb:
        raise ConfigurationError(
            f"tokenizer vocab {n_tok} 大于模型 embedding {n_emb}"
        )
    if sid_table is not None:
        missing = [
            tok
            for tok in sid_table.all_tokens()
            if tokenizer.convert_tokens_to_ids(tok) in (None, tokenizer.unk_token_id)
        ]
        if missing:
            raise ConfigurationError(
                f"SID token 不在 tokenizer 词表里（例：{missing[:3]}）"
            )


def load_trained(
    checkpoint_dir: str | Path,
    *,
    dtype: str = "fp32",
    local_rank: int = 0,
    attn_implementation: str | None = None,
    backbone: str | Path | None = None,
    sid_table: SidTable | None = None,
) -> SidModelBundle:
    """加载我们自己存的全参 checkpoint（SID token 已经在里面了）。

    RL 接 SFT、以及单独评测走这条路 —— 不再有 ``PeftModel.from_pretrained``
    那套 adapter 合并流程。
    """
    path = Path(checkpoint_dir)
    if not path.is_dir():
        raise MissingArtifactError(f"checkpoint 目录不存在：{path}")

    tokenizer = _load_tokenizer_for_checkpoint(
        path, backbone=backbone, sid_table=sid_table
    )
    load_kwargs: dict[str, Any] = {"dtype": resolve_dtype(dtype)}
    if attn_implementation:
        load_kwargs["attn_implementation"] = str(attn_implementation)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    except Exception:
        load_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    _assert_tokenizer_matches_model(tokenizer, model, sid_table=sid_table)
    if torch.cuda.is_available():
        model = model.to(f"cuda:{local_rank}")
    new_ids = []
    if sid_table is not None:
        new_ids = [tokenizer.convert_tokens_to_ids(t) for t in sid_table.all_tokens()]
    return SidModelBundle(
        tokenizer=tokenizer, model=model, new_token_ids=new_ids, n_new_tokens=0
    )
