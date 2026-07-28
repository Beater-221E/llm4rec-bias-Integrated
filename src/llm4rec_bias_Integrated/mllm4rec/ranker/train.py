# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
# (train_ranker.py + trainer/llm.py — LoRA ranker)
#
# Engineering: V100 defaults to float16 (not bf16); model via AutoModelForCausalLM.

"""Train multimodal LLM ranker with LoRA on letter candidates."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from llm4rec_bias_Integrated.mllm4rec.ranker.data import load_ranker_bundle

logger = logging.getLogger("llm4rec_bias_Integrated.mllm4rec.ranker")


@dataclass
class RankerConfig:
    dataset_pkl: Path
    retrieved_pkl: Path
    export_root: Path
    llm_base_model: str
    llm_base_tokenizer: str | None = None
    device: str = "cuda"
    seed: int = 42
    load_in_4bit: bool = False  # V100: prefer fp16 LoRA without bnb by default
    dtype: str = "float16"
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_num_epochs: int = 3
    lora_lr: float = 1e-4
    lora_micro_batch_size: int = 2
    val_batch_size: int = 2
    test_batch_size: int = 2
    llm_negative_sample_size: int = 19
    llm_max_history: int = 25
    llm_max_title_len: int = 32
    llm_max_text_len: int = 1536
    max_train_steps: int | None = None  # smoke override
    grad_clip: float = 1.0


def _letter_token_ids(tokenizer, num_classes: int) -> list[int]:
    ids = []
    for i in range(num_classes):
        letter = chr(ord("A") + i)
        tid = tokenizer.encode(letter, add_special_tokens=False)
        if not tid:
            tid = tokenizer.encode(" " + letter, add_special_tokens=False)
        ids.append(tid[-1])
    return ids


def _score_letters(logits_last: torch.Tensor, letter_ids: list[int]) -> torch.Tensor:
    # logits_last: [B, vocab]
    cols = torch.tensor(letter_ids, device=logits_last.device)
    return logits_last.index_select(-1, cols)


def train_ranker(cfg: RankerConfig) -> Path:
    from llm4rec_bias_Integrated.tracking.inplace_progress import (
        install_inplace_progress,
        write_progress_status,
    )

    install_inplace_progress(force=True)
    write_progress_status("ranker: loading…")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.manual_seed(cfg.seed)

    tok_name = cfg.llm_base_tokenizer or cfg.llm_base_model
    bundle = load_ranker_bundle(
        dataset_pkl=cfg.dataset_pkl,
        retrieved_pkl=cfg.retrieved_pkl,
        tokenizer_name=tok_name,
        llm_negative_sample_size=cfg.llm_negative_sample_size,
        llm_max_history=cfg.llm_max_history,
        llm_max_title_len=cfg.llm_max_title_len,
        llm_max_text_len=cfg.llm_max_text_len,
        lora_micro_batch_size=cfg.lora_micro_batch_size,
        val_batch_size=cfg.val_batch_size,
        test_batch_size=cfg.test_batch_size,
        seed=cfg.seed,
    )
    tokenizer = bundle["tokenizer"]
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    num_classes = bundle["num_classes"]
    letter_ids = _letter_token_ids(tokenizer, num_classes)

    torch_dtype = torch.float16 if cfg.dtype == "float16" else torch.bfloat16
    if cfg.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_base_model,
            quantization_config=bnb,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_base_model,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        model.to(cfg.device)

    model.gradient_checkpointing_enable()
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lora_lr
    )
    export = Path(cfg.export_root)
    export.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_ndcg = -1.0
    best_dir = export / "best_lora"

    for epoch in range(cfg.lora_num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ranker epoch {epoch+1}")
        for batch in pbar:
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            if cfg.max_train_steps and global_step >= cfg.max_train_steps:
                break
        # validate letter ranking NDCG@10
        ndcg10 = _eval_letter_ndcg(model, val_loader, cfg.device, letter_ids, k=10)
        logger.info("epoch %s val NDCG@10=%.4f", epoch + 1, ndcg10)
        if ndcg10 >= best_ndcg:
            best_ndcg = ndcg10
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        if cfg.max_train_steps and global_step >= cfg.max_train_steps:
            break

    if best_dir.is_dir():
        # reload best for test
        pass
    subset_metrics = _eval_letter_metrics(
        model, test_loader, cfg.device, letter_ids, ks=[1, 5, 10]
    )
    report = {
        "subset_metrics": subset_metrics,
        "test_retrieval": {
            k: v
            for k, v in bundle["test_retrieval"].items()
            if k != "original_metrics" or True
        },
        "best_val_ndcg10": best_ndcg,
    }
    # make retrieval metrics JSON-serializable
    def _ser(o):
        if isinstance(o, dict):
            return {k: _ser(v) for k, v in o.items()}
        if isinstance(o, (float, int, str, bool)) or o is None:
            return o
        return float(o) if hasattr(o, "item") else str(o)

    (export / "subset_metrics.json").write_text(
        json.dumps(_ser(report), indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", export / "subset_metrics.json")
    return export


@torch.no_grad()
def _eval_letter_ndcg(model, loader, device, letter_ids, k=10) -> float:
    model.eval()
    scores_all = []
    labels_all = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]
        scores = _score_letters(logits, letter_ids)
        scores_all.append(scores.cpu())
        labels_all.append(labels.cpu())
    if not scores_all:
        return 0.0
    scores_t = torch.cat(scores_all, dim=0)
    labels_t = torch.cat(labels_all, dim=0)
    from llm4rec_bias_Integrated.mllm4rec.metrics import absolute_recall_mrr_ndcg_for_ks

    m = absolute_recall_mrr_ndcg_for_ks(scores_t, labels_t, [k])
    return float(m.get(f"NDCG@{k}", 0.0))


@torch.no_grad()
def _eval_letter_metrics(model, loader, device, letter_ids, ks):
    model.eval()
    scores_all, labels_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]
        scores_all.append(_score_letters(logits, letter_ids).cpu())
        labels_all.append(labels.cpu())
    if not scores_all:
        return {}
    from llm4rec_bias_Integrated.mllm4rec.metrics import absolute_recall_mrr_ndcg_for_ks

    return absolute_recall_mrr_ndcg_for_ks(
        torch.cat(scores_all, 0), torch.cat(labels_all, 0), ks
    )
