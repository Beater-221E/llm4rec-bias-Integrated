"""框架核心：配置组合、路径、异常、复现性。

这里刻意【不做】任何重导入 —— ``core.exceptions`` / ``core.paths`` 要能在
不装 omegaconf / torch 的环境里单独 import（纯指标计算、离线分析脚本都用得上）。
需要配置组合就显式 ``from llm4rec.core.compose import compose``。
"""
