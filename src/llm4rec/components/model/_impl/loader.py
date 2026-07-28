"""Causal LM loader (CUDA only)."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from llm4rec.core.exceptions import CheckpointError
from llm4rec.components.model._impl.base import PrecisionConfig, require_cuda, resolve_precision
from llm4rec.components.model._impl.peft import apply_lora, load_and_merge_adapter
from llm4rec.components.model._impl.quantization import build_bnb_config
from llm4rec.components.model._impl.tokenizer import load_tokenizer


def load_causal_lm(
    checkpoint: str,
    *,
    precision: PrecisionConfig | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = False,
    quantization: dict[str, Any] | None = None,
    use_flash_attention: bool = False,
    local_rank: int = 0,
) -> PreTrainedModel:
    require_cuda()
    precision = precision or resolve_precision("auto")
    torch_device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda"

    kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    }
    bnb = build_bnb_config(quantization)
    if bnb is not None:
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = {"": torch_device}
    else:
        kwargs["dtype"] = precision.dtype

    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
    except TypeError:
        # older transformers used torch_dtype=
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = precision.dtype
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(f"Failed to load checkpoint '{checkpoint}': {exc}") from exc

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if bnb is None:
        model = model.to(torch_device)
    return model


def load_model_bundle(
    model_cfg: dict[str, Any],
    peft_cfg: dict[str, Any] | None = None,
    *,
    adapter_path: str | None = None,
    sft_adapter_path: str | None = None,
    for_training: bool = False,
    local_rank: int = 0,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, PrecisionConfig]:
    """Load tokenizer + model on CUDA, optionally merge adapters / apply LoRA."""
    require_cuda()
    checkpoint = model_cfg.get("checkpoint")
    if not checkpoint:
        raise CheckpointError("model.checkpoint is required")
    precision = resolve_precision(str(model_cfg.get("dtype") or "auto"))
    tok = load_tokenizer(
        str(checkpoint),
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    model = load_causal_lm(
        str(checkpoint),
        precision=precision,
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False))
        and for_training,
        quantization=model_cfg.get("quantization"),
        use_flash_attention=bool(model_cfg.get("use_flash_attention", False)),
        local_rank=local_rank,
    )
    for path in (sft_adapter_path, adapter_path):
        if path:
            model = load_and_merge_adapter(model, path)

    if for_training and peft_cfg and peft_cfg.get("enabled", True):
        model = apply_lora(model, peft_cfg)

    return tok, model, precision
