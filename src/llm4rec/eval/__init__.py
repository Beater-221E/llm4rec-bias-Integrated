"""统一评测层：ranked-list 指标 + bias 指标 + 训练中在线评测。

三条路线（minionerec / recr1 / dpo4rec）共用这里的全部实现。
路线之间的差异只体现在 ``llm4rec.decoders`` 里各自的 Decoder。

``bias`` 和 ``catalog`` 是纯 numpy 的，不依赖 torch —— 离线复算指标时
可以只 import 它们；``online`` 依赖 torch/transformers，按需显式导入。
"""

from llm4rec.eval.bias import RankedResult, bias_delta, compute_bias_metrics
from llm4rec.eval.catalog import ItemCatalog

__all__ = ["ItemCatalog", "RankedResult", "bias_delta", "compute_bias_metrics"]
