"""Conservative automatic precision selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llm4rec.runtime.hardware import HardwareInfo


@dataclass
class PrecisionChoice:
    precision: str  # fp32 | fp16 | bf16
    amp: bool
    grad_scaler: bool
    source: str  # auto | override

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_precision(
    requested: str | None,
    hw: HardwareInfo,
    *,
    route: str = "",
    allow_fp16: bool | None = None,
) -> PrecisionChoice:
    """Select training precision.

    Policy:
      * CUDA unavailable → fp32
      * CUDA + BF16 supported → bf16 mixed
      * CUDA without BF16 → fp16 AMP when route validated, else fp32
      * FP8 is never chosen automatically
    """
    req = (requested or "auto").lower().strip()
    if req in {"fp8", "float8"}:
        raise ValueError("FP8 is opt-in only and not auto-selected; unset hardware.precision=fp8")

    if req not in {"auto", "fp32", "fp16", "float16", "bf16", "bfloat16"}:
        raise ValueError(f"unsupported precision '{requested}'")

    if req != "auto":
        prec = "bf16" if req in {"bf16", "bfloat16"} else ("fp16" if req in {"fp16", "float16"} else "fp32")
        if prec == "bf16" and hw.cuda_available and not hw.bf16_supported:
            # Fall back rather than crash on V100-class hardware
            prec = "fp16" if (allow_fp16 is not False) else "fp32"
            return PrecisionChoice(
                precision=prec,
                amp=prec == "fp16",
                grad_scaler=prec == "fp16",
                source="override->fallback",
            )
        return PrecisionChoice(
            precision=prec,
            amp=prec == "fp16",
            grad_scaler=prec == "fp16",
            source="override",
        )

    if not hw.cuda_available:
        return PrecisionChoice(precision="fp32", amp=False, grad_scaler=False, source="auto")

    # Prefer capability (cc>=8) in addition to torch.cuda.is_bf16_supported().
    # Some builds report bf16_supported=True on Volta even though hardware bf16 is absent.
    cc_ok = hw.compute_capability is not None and hw.compute_capability[0] >= 8
    if hw.bf16_supported and cc_ok:
        return PrecisionChoice(precision="bf16", amp=False, grad_scaler=False, source="auto")

    # No BF16 (e.g. V100). MiniOneRec SID SFT historically unstable in fp16 → fp32 default.
    # Rec-R1 / DPO4Rec without new SID embeddings may use fp16 AMP.
    if allow_fp16 is None:
        allow_fp16 = route in {"recr1", "dpo4rec"}
    if allow_fp16:
        return PrecisionChoice(precision="fp16", amp=True, grad_scaler=True, source="auto")
    return PrecisionChoice(precision="fp32", amp=False, grad_scaler=False, source="auto")
