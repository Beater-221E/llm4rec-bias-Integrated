"""MiniOneRec trainers — wrap SID SFT/GRPO implementations."""

from __future__ import annotations

from llm4rec.components.trainer._impl.sid_grpo import SidGRPOTrainer
from llm4rec.components.trainer._impl.sid_sft import SidSFTTrainer

# Prefer workflow-local copies if present (same code, clearer ownership)
try:
    from llm4rec.workflows.minionerec._sid_sft import SidSFTTrainer as _LocalSFT
    from llm4rec.workflows.minionerec._sid_grpo import SidGRPOTrainer as _LocalGRPO

    SidSFTTrainer = _LocalSFT
    SidGRPOTrainer = _LocalGRPO
except Exception:  # noqa: BLE001
    pass

__all__ = ["SidSFTTrainer", "SidGRPOTrainer"]
