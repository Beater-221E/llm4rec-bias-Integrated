# Adapted from:
# https://github.com/wangyuxiang123/MLLM4Rec
#
# Original behavior is preserved unless explicitly documented.

"""BLIP2 poster captioning (official ``process_item_blip2`` semantics)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("llm4rec_bias_Integrated.mllm4rec")

CaptionMode = Literal["original", "batched"]


def load_caption_cache(path: Path) -> dict[int, str]:
    cache: dict[int, str] = {}
    if not path.is_file():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[int(row["internal_item_id"])] = str(row.get("caption") or "")
    return cache


def append_caption(path: Path, *, internal_item_id: int, caption: str, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "internal_item_id": int(internal_item_id),
                    "caption": caption,
                    "status": status,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _torch_dtype(name: str):
    import torch

    key = name.lower()
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def generate_captions_for_dataset(
    dataset: dict[str, Any],
    *,
    img_dir: Path,
    model_name_or_path: str,
    device: str = "cuda",
    dtype: str = "float16",
    mode: CaptionMode = "original",
    batch_size: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    max_items: int | None = None,
    captions_path: Path | None = None,
    start_index: int = 0,
) -> dict[int, str]:
    """Fill ``dataset['meta_img_des']`` with BLIP2 captions (or ``\"\"`` if no image).

    Original mode mirrors official calls:
    - ``Blip2Processor.from_pretrained`` **inside** the per-item loop
    - ``Image.open`` without ``.convert(\"RGB\")``
    - ``model.generate(**inputs)`` with no extra kwargs
    """
    import torch
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    img_dir = Path(img_dir)
    captions_path = captions_path or (img_dir.parent / "captions.jsonl")
    cache = load_caption_cache(captions_path) if (resume and not overwrite) else {}
    if overwrite and captions_path.is_file():
        captions_path.unlink()
        cache = {}

    meta: dict[Any, str] = dataset["meta"]
    keys = list(meta.keys())
    if start_index:
        keys = keys[start_index:]
    if max_items is not None:
        keys = keys[: max(0, int(max_items))]

    torch_dtype = _torch_dtype(dtype)
    logger.info("Loading BLIP2 from %s (dtype=%s, device=%s)", model_name_or_path, dtype, device)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype
    )
    model.to(device)
    model.eval()

    meta_img_des: dict[int, str] = dict(dataset.get("meta_img_des") or {})
    # Ensure all existing meta keys remain present when resuming partial runs
    for k in meta:
        meta_img_des.setdefault(int(k), cache.get(int(k), ""))

    processor_shared = Blip2Processor.from_pretrained(model_name_or_path)
    # Official script reloads the processor inside the loop; that is extremely slow
    # on cold/HF-cache loads. We keep generate(**inputs) with no extra kwargs
    # (original semantics) and reuse one processor. Set mode=batched for the same
    # path with an explicit batch_size knob.
    if mode == "batched":
        batch_size = max(1, int(batch_size))
    else:
        batch_size = 1

    pending: list[int] = []
    for key in keys:
        iid = int(key)
        if resume and not overwrite and iid in cache:
            meta_img_des[iid] = cache[iid]
            continue
        pending.append(iid)

    def _caption_one(iid: int, processor) -> str:
        img_path = img_dir / f"{iid}.jpg"
        if not img_path.is_file():
            return ""
        try:
            image = Image.open(img_path)  # official: no convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device, torch_dtype)
            with torch.inference_mode():
                generated_ids = model.generate(**inputs)
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as exc:  # noqa: BLE001 — keep item; empty caption
            logger.warning("Caption failed for item %s: %s", iid, exc)
            return ""

    done = 0
    from tqdm import tqdm

    for start in tqdm(
        range(0, len(pending), batch_size),
        desc="BLIP2 captions",
        unit="batch",
        dynamic_ncols=True,
        total=(len(pending) + batch_size - 1) // batch_size if pending else 0,
    ):
        chunk = pending[start : start + batch_size]
        for iid in chunk:
            caption = _caption_one(iid, processor_shared)
            status = (
                "ok"
                if caption
                else (
                    "missing_image"
                    if not (img_dir / f"{iid}.jpg").is_file()
                    else "empty"
                )
            )
            meta_img_des[iid] = caption
            append_caption(
                captions_path, internal_item_id=iid, caption=caption, status=status
            )
            done += 1

    # Official invariant: every meta key has a caption entry
    for k in meta:
        meta_img_des.setdefault(int(k), "")
    dataset["meta_img_des"] = {int(k): meta_img_des[int(k)] for k in meta}
    assert set(dataset["meta"].keys()) == set(dataset["meta_img_des"].keys())
    return dataset["meta_img_des"]
