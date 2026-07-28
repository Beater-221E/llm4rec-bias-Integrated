"""MLLM4Rec workflow — multimodal Retriever + Ranker (behavior-preserving)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.components.dataset.base import DatasetBundle
from llm4rec.core.context import ExperimentContext
from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.paths import project_root
from llm4rec.workflows.base import BaseWorkflow
from llm4rec.workflows.registry import register_workflow


@register_workflow("mllm4rec")
class MLLM4RecWorkflow(BaseWorkflow):
    """Official MLLM4Rec pipeline: multimodal data → LRU retriever → LLM ranker.

    This replaces the previous text-only stub that subclassed GRPO4Rec.
    Algorithm code lives in ``retriever.py`` / ``ranker.py`` / ``data/``.
    """

    name = "mllm4rec"

    def __init__(self, context: ExperimentContext | None = None, **kwargs: Any) -> None:
        super().__init__(context, **kwargs)
        self._dataset_pkl: Path | None = None
        self._retrieved_pkl: Path | None = None
        self._train_summary: dict[str, Any] = {}

    def required_stages(self) -> list[str]:
        return [
            "prepare_data",
            "train_retriever",
            "train_ranker",
            "evaluate",
        ]

    def prepare_data(self) -> DatasetBundle:
        if self.context is None:
            raise ConfigurationError("context required")
        from llm4rec.workflows.mllm4rec.data.config import load_data_config
        from llm4rec.workflows.mllm4rec.data.dataset_factory import dataset_factory
        from llm4rec.workflows.mllm4rec.data.serializer import load_pickle

        cfg = self.context.config
        data_cfg_path = (cfg.get("mllm4rec") or {}).get("data_config")
        if data_cfg_path:
            path = Path(data_cfg_path)
            if not path.is_absolute():
                path = project_root() / path
            data_cfg = load_data_config(path)
        else:
            # Default ml-100k official config
            path = project_root() / "configs" / "dataset" / "mllm4rec_ml100k.yaml"
            data_cfg = load_data_config(path)

        ds = dataset_factory(data_cfg)
        # Build only if dataset.pkl missing
        out_pkl = Path(data_cfg.output_dir) / "dataset.pkl"
        if not out_pkl.is_absolute():
            out_pkl = project_root() / out_pkl
        if not out_pkl.is_file():
            ds.build()
        payload = load_pickle(out_pkl)
        self._dataset_pkl = out_pkl

        users = list(payload.get("user_id_map", {}).keys()) if isinstance(payload, dict) else []
        items = list(payload.get("item_id_map", {}).keys()) if isinstance(payload, dict) else []
        bundle = DatasetBundle(
            name=str(getattr(data_cfg, "dataset_code", "mllm4rec")),
            interactions=[],
            users=[str(u) for u in users],
            items=[str(i) for i in items],
            sequences={},
            metadata={"dataset_pkl": str(out_pkl)},
            extras={
                "images": payload.get("meta", {}).get("image_path") if isinstance(payload, dict) else None,
                "captions": True,
                "candidates": True,
                "official_payload": payload,
            },
        )
        self._bundle = bundle
        return bundle

    def build_model(self) -> Any:
        # Retriever (LRU) and Ranker (Qwen+LoRA) are built inside their train stages
        return {"retriever": "lru", "ranker": "qwen_lora"}

    def train(self) -> dict[str, Any]:
        if self.context is None:
            raise ConfigurationError("context required")
        from llm4rec.workflows.mllm4rec.retriever import train_retriever_from_config
        from llm4rec.workflows.mllm4rec.ranker import train_ranker_from_config

        summary: dict[str, Any] = {}
        mllm_cfg = self.context.config.get("mllm4rec") or {}
        retriever_cfg = mllm_cfg.get("retriever_config") or "mllm4rec_retriever"
        ranker_cfg = mllm_cfg.get("ranker_config") or "mllm4rec_ranker"

        ret = train_retriever_from_config(retriever_cfg, dataset_pkl=self._dataset_pkl)
        summary["retriever"] = ret
        self._retrieved_pkl = Path(ret["retrieved_pkl"]) if ret.get("retrieved_pkl") else None

        rank = train_ranker_from_config(
            ranker_cfg,
            retrieved_pkl=self._retrieved_pkl,
            dataset_pkl=self._dataset_pkl,
        )
        summary["ranker"] = rank
        self._train_summary = summary
        return summary

    def evaluate(self) -> dict[str, Any]:
        metrics = {}
        for stage in ("retriever", "ranker"):
            part = (self._train_summary or {}).get(stage) or {}
            if "metrics" in part:
                metrics[stage] = part["metrics"]
        return metrics

    def inference(self, user_seq: list[int], **kwargs: Any) -> Any:
        raise NotImplementedError("Use trained retriever/ranker checkpoints for inference")

    # Minimal CLI compatibility (legacy train path should use mllm4rec.sh)
    def build_examples(self, dataset: Any, split: str) -> list:
        return []

    def build_examples_with_spec(self, dataset: Any, split: str, task_spec: Any) -> list:
        return []

    def set_examples(self, train: list, eval_examples: list) -> None:
        return None

    def build_model_legacy(self, context: ExperimentContext) -> Any:
        self.bind(context)
        return self.build_model()

    def build_trainer(self, context: ExperimentContext) -> Any:
        self.bind(context)
        raise ConfigurationError(
            "MLLM4Rec training uses retriever/ranker stages via mllm4rec.sh "
            "or MLLM4RecWorkflow.train(); not the letter SFT trainer."
        )

    def build_evaluator(self, context: ExperimentContext) -> Any:
        self.bind(context)
        return self

    def build_probes(self, context: ExperimentContext) -> list[Any]:
        return []
