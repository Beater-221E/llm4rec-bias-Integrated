"""Re-export distributed plan helpers for workflows."""

from llm4rec.components.trainer._impl.distributed import (
    resolve_distributed_plan,
)


def resolve_plan(context):
    return resolve_distributed_plan(
        context.config.get("training") or {},
        model_name=context.model_name,
    )


__all__ = ["resolve_plan", "resolve_distributed_plan"]
