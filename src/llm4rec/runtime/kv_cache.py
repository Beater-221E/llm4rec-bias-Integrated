"""Generation cache helpers (static vs dynamic KV)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class KVCacheChoice:
    use_cache: bool
    requested: str
    effective: str  # static | dynamic | disabled
    cache_implementation: str | None = None
    fallback_reason: str | None = None

    def fallback_to_dynamic(self, reason: str) -> None:
        """Mutate after a failed static-cache generate attempt."""
        self.effective = "dynamic"
        self.cache_implementation = None
        self.fallback_reason = str(reason)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_kv_cache(
    cfg: dict[str, Any],
    *,
    constrained: bool = False,
) -> KVCacheChoice:
    """Resolve KV cache semantics.

    ``cache: auto`` → prefer dynamic (broad compat); ``static`` attempts static
    where supported. Constrained MiniOneRec generation stays dynamic unless a
    validated static path is explicitly forced later.
    """
    gen = ((cfg.get("optimization") or {}).get("generation") or {})
    requested = str(gen.get("cache") or "auto").lower()
    if requested in {"false", "off", "disabled", "none"}:
        choice = KVCacheChoice(False, requested, "disabled", None, None)
    elif constrained:
        # MiniOneRec constrained beam + prefix_allowed_tokens_fn: keep dynamic.
        choice = KVCacheChoice(True, requested, "dynamic", None, "constrained_generation")
    elif requested == "static":
        try:
            from transformers.cache_utils import StaticCache  # noqa: F401

            choice = KVCacheChoice(True, requested, "static", "static", None)
        except Exception as exc:  # noqa: BLE001
            choice = KVCacheChoice(
                True, requested, "dynamic", None, f"static_cache_unavailable:{exc}"
            )
    else:  # auto / dynamic
        choice = KVCacheChoice(True, requested, "dynamic", None, None)
    gen["_effective_cache"] = choice.effective
    gen["_cache_choice"] = choice.to_dict()
    return choice


def persist_kv_choice(cfg: dict[str, Any], choice: KVCacheChoice) -> None:
    """Write the (possibly mutated) choice back into cfg for the execution manifest."""
    gen = ((cfg.setdefault("optimization", {})).setdefault("generation", {}))
    gen["_effective_cache"] = choice.effective
    gen["_cache_choice"] = choice.to_dict()
    if choice.fallback_reason:
        gen["_cache_fallback_reason"] = choice.fallback_reason


def generation_cache_kwargs(choice: KVCacheChoice) -> dict[str, Any]:
    """kwargs for HF ``generate``. Never claims static unless truly requested+supported."""
    if not choice.use_cache:
        return {"use_cache": False}
    kwargs: dict[str, Any] = {"use_cache": True}
    if choice.effective == "static" and choice.cache_implementation:
        kwargs["cache_implementation"] = choice.cache_implementation
    return kwargs
