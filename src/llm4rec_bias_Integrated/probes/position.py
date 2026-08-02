# 兼容层：转发 position probe，并显式导出测试依赖的 _kendall_tau（star import 不会带出下划线名）
from llm4rec.components.evaluation.probes.position import *  # noqa: F403
from llm4rec.components.evaluation.probes.position import _kendall_tau  # noqa: F401
