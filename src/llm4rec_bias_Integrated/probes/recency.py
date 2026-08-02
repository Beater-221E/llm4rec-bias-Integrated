# 兼容层：转发 recency probe，并显式导出测试依赖的 _apply_intervention
from llm4rec.components.evaluation.probes.recency import *  # noqa: F403
from llm4rec.components.evaluation.probes.recency import _apply_intervention  # noqa: F401
