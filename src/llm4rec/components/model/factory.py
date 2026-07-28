"""ModelFactory — sole entrypoint for loading Qwen / LoRA / quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm4rec.components.model.lora import apply_lora_config, merge_adapter
from llm4rec.components.model.qwen import QWEN_ALIASES, resolve_qwen_checkpoint
from llm4rec.components.model.utils import resolve_precision_name
from llm4rec.components.model._impl.base import PrecisionConfig
from llm4rec.components.model._impl.loader import load_causal_lm, load_model_bundle
from llm4rec.core.exceptions import CheckpointError, ConfigurationError


@dataclass
class ModelBundle:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    precision: PrecisionConfig
    checkpoint: str
    name: str


class ModelFactory:
    """Unified model construction for all workflows.

    Workflows must not call ``transformers.from_pretrained`` directly.
    """

    SUPPORTED = tuple(QWEN_ALIASES.keys())

    @classmethod
    def resolve_checkpoint(cls, model_cfg: dict[str, Any]) -> str:
        name = str(model_cfg.get("name") or "")
        ckpt = model_cfg.get("checkpoint")
        if ckpt:
            return str(ckpt)
        resolved = resolve_qwen_checkpoint(name)
        if not resolved:
            raise CheckpointError(
                f"Cannot resolve checkpoint for model.name={name!r}. "
                f"Known aliases: {', '.join(cls.SUPPORTED)}"
            )
        return resolved

    @classmethod
    def from_config(
        cls,
        model_cfg: dict[str, Any],
        peft_cfg: dict[str, Any] | None = None,
        *,
        adapter_path: str | None = None,
        sft_adapter_path: str | None = None,
        for_training: bool = False,
        local_rank: int = 0,
    ) -> ModelBundle:
        cfg = dict(model_cfg)
        cfg["checkpoint"] = cls.resolve_checkpoint(cfg)
        if "dtype" in cfg:
            cfg["dtype"] = resolve_precision_name(str(cfg["dtype"]))
        tok, model, precision = load_model_bundle(
            cfg,
            peft_cfg,
            adapter_path=adapter_path,
            sft_adapter_path=sft_adapter_path,
            for_training=for_training,
            local_rank=local_rank,
        )
        return ModelBundle(
            tokenizer=tok,
            model=model,
            precision=precision,
            checkpoint=str(cfg["checkpoint"]),
            name=str(cfg.get("name") or cfg["checkpoint"]),
        )

    @classmethod
    def load_base(
        cls,
        checkpoint: str,
        *,
        dtype: str = "auto",
        gradient_checkpointing: bool = False,
        quantization: dict[str, Any] | None = None,
        use_flash_attention: bool = False,
        local_rank: int = 0,
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> PreTrainedModel:
        """Load a causal LM only (no tokenizer / LoRA)."""
        precision = PrecisionConfig  # placate type checkers below
        _ = precision
        from llm4rec.components.model._impl.base import resolve_precision

        return load_causal_lm(
            checkpoint,
            precision=resolve_precision(dtype),
            revision=revision,
            trust_remote_code=trust_remote_code,
            gradient_checkpointing=gradient_checkpointing,
            quantization=quantization,
            use_flash_attention=use_flash_attention,
            local_rank=local_rank,
        )

    @classmethod
    def apply_lora(cls, model: PreTrainedModel, peft_cfg: dict[str, Any]) -> PreTrainedModel:
        return apply_lora_config(model, peft_cfg)

    @classmethod
    def merge_adapter(cls, model: PreTrainedModel, adapter_path: str) -> PreTrainedModel:
        return merge_adapter(model, adapter_path)

    @classmethod
    def ensure_no_direct_hf(cls, caller: str) -> None:
        """Documentation hook — raise if a workflow bypasses the factory."""
        raise ConfigurationError(
            f"{caller} must load models via ModelFactory, not transformers.from_pretrained"
        )
