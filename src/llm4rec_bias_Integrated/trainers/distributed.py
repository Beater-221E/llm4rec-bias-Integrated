# 兼容层：转发 distributed helpers，并显式导出测试 monkeypatch 用的 _rendezvous_dir
from llm4rec.components.trainer._impl.distributed import *  # noqa: F403
from llm4rec.components.trainer._impl.distributed import _rendezvous_dir  # noqa: F401
