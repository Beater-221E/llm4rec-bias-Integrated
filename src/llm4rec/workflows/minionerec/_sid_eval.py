"""Full-catalog SID evaluation (constrained beam + free-gen validity)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel

from llm4rec.compatibility.llm4rec_bias_eval import (
    catalog_pop_lift,
    coverage_at,
    delta_gap,
    gini,
    ips_weight,
    ndcg_at,
    popularity_tier,
    snips,
)
from llm4rec.core.exceptions import MissingArtifactError
from llm4rec.core.reproducibility import write_json
from llm4rec.core.schemas import EvaluationResult
from llm4rec.components.model._impl.base import require_cuda
from llm4rec.components.model._impl.sid import prepare_sid_model
from llm4rec.workflows.minionerec.semantic_ids.build import load_sid_jsonl, sid_dir
from llm4rec.workflows.minionerec.semantic_ids.table import SidTable


@torch.no_grad()
def beam_retrieve(tok, model, device, table: SidTable, messages: list[dict], k: int) -> list[str]:
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    ids = (enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc).to(device)
    eos = tok.eos_token_id
    out = model.generate(
        ids,
        max_new_tokens=table.levels + 1,
        num_beams=k,
        num_return_sequences=k,
        prefix_allowed_tokens_fn=table.prefix_fn(tok, ids.shape[1], eos),
        early_stopping=True,
        do_sample=False,
        pad_token_id=eos,
    )
    items: list[str] = []
    for seq in out:
        text = tok.decode(seq[ids.shape[1] :], skip_special_tokens=False)
        item = table.parse(text)
        if item is not None and item not in items:
            items.append(item)
    return items


@torch.no_grad()
def free_top1_valid(tok, model, device, table: SidTable, messages: list[dict]) -> bool:
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    ids = (enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc).to(device)
    out = model.generate(
        ids,
        max_new_tokens=table.levels + 2,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=False)
    return table.parse(text) is not None


def evaluate_sid_checkpoint(
    *,
    model_cfg: dict[str, Any],
    processed_dir: Path,
    adapter_path: str | None,
    sft_adapter_path: str | None = None,
    split: str = "test",
    top_k: int = 10,
    max_examples: int | None = 300,
    free_gen_n: int = 50,
    ips_gamma: float = 1.0,
    predictions_dir: Path | None = None,
) -> EvaluationResult:
    require_cuda()
    out_sid = sid_dir(processed_dir)
    table_path = out_sid / "semantic_ids.json"
    meta_path = out_sid / "item_meta.json"
    data_path = out_sid / f"sid_{split}.jsonl"
    for p in (table_path, meta_path, data_path):
        if not p.is_file():
            raise MissingArtifactError(f"missing {p}")

    table = SidTable(table_path)
    tok, model, _ = prepare_sid_model(
        str(model_cfg["checkpoint"]),
        table,
        dtype=str(model_cfg.get("dtype") or "auto"),
    )
    if sft_adapter_path:
        model = PeftModel.from_pretrained(model, sft_adapter_path).merge_and_unload()
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()
    device = next(model.parameters()).device

    with meta_path.open(encoding="utf-8") as f:
        import json

        meta = {str(k): v for k, v in json.load(f).items()}
    pop_mean = float(np.mean([float(m["pop_quantile"]) for m in meta.values()]))
    rows = load_sid_jsonl(data_path, limit=max_examples)

    hr1, hrk, ndcgs, lifts, gaps = [], [], [], [], []
    exposure: Counter[str] = Counter()
    tier_hits: dict[str, list[float]] = {"head": [], "mid": [], "tail": []}
    ips_w, ips_hit_w, ips_ndcg_w = [], [], []
    catalog_n = len(table.codes)

    for r in rows:
        items = beam_retrieve(tok, model, device, table, r["prompt"], top_k)
        tgt = str(r["target_item"])
        rank = items.index(tgt) if tgt in items else None
        hit = rank is not None
        dcg = ndcg_at(rank, top_k) if hit else 0.0
        hr1.append(float(rank == 0))
        hrk.append(float(hit))
        ndcgs.append(float(dcg))
        for it in items:
            exposure[it] += 1
        top1 = items[0] if items else None
        if top1 is not None and top1 in meta:
            q = float(meta[top1]["pop_quantile"])
            lifts.append(catalog_pop_lift(q, pop_mean))
            hist_mean = float(r.get("hist_pop_mean", 0.5))
            gaps.append(delta_gap(q, hist_mean))
        tgt_q = float(meta.get(tgt, {}).get("pop_quantile", 0.5))
        tier = popularity_tier(tgt_q)
        tier_hits[tier].append(float(hit))
        w = ips_weight(int(meta.get(tgt, {}).get("count", 1)), ips_gamma)
        ips_w.append(w)
        ips_hit_w.append(w * float(hit))
        ips_ndcg_w.append(w * float(dcg))

    free = []
    for r in rows[: free_gen_n]:
        free.append(float(free_top1_valid(tok, model, device, table, r["prompt"])))

    counts = np.zeros(catalog_n, dtype=np.float64)
    id_list = list(table.codes.keys())
    id_index = {i: n for n, i in enumerate(id_list)}
    for it, c in exposure.items():
        if it in id_index:
            counts[id_index[it]] = c

    metrics: dict[str, float | str] = {
        "n": float(len(rows)),
        "hr@1": float(np.mean(hr1)) if hr1 else 0.0,
        f"hr@{top_k}": float(np.mean(hrk)) if hrk else 0.0,
        f"ndcg@{top_k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "pop_lift@1": float(np.mean(lifts)) if lifts else 0.0,
        "delta_gap": float(np.mean(gaps)) if gaps else 0.0,
        "exposure_gini": gini(counts),
        f"coverage@{top_k}": coverage_at(counts),
        "free_gen_valid_rate": float(np.mean(free)) if free else 0.0,
        f"hr_ips@{top_k}": snips(ips_w, ips_hit_w) or 0.0,
        f"ndcg_ips@{top_k}": snips(ips_w, ips_ndcg_w) or 0.0,
    }
    for tier, vals in tier_hits.items():
        metrics[f"hr@{top_k}_{tier}"] = float(np.mean(vals)) if vals else 0.0

    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            predictions_dir / f"sid_{split}_summary.json",
            {"metrics": metrics, "n": len(rows)},
        )

    return EvaluationResult(
        metrics=metrics,
        slices={t: {"n": len(v), f"hr@{top_k}": float(np.mean(v)) if v else 0.0} for t, v in tier_hits.items()},
        predictions_path=None,
        metadata={"route": "sid", "top_k": top_k, "catalog_n": catalog_n},
    )
