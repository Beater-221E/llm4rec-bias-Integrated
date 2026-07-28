"""Dataset validator / multimodal coverage reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm4rec_bias_Integrated.data.mllm4rec.compatibility import validate_official_schema
from llm4rec_bias_Integrated.data.mllm4rec.serializer import load_pickle
from llm4rec_bias_Integrated.data.mllm4rec.tmdb_client import load_match_cache


def validate_dataset_pkl(
    path: str | Path,
    *,
    require_captions: bool = False,
) -> dict[str, Any]:
    """Validate pickle + optional TMDb/poster/caption coverage."""
    path = Path(path)
    root = path.parent
    dataset = load_pickle(path)
    errors = validate_official_schema(dataset, require_captions=require_captions)

    meta = dataset.get("meta") or {}
    img_dir = root / "img"
    matches = load_match_cache(root / "tmdb_matches.jsonl")
    captions = dataset.get("meta_img_des") or {}

    downloaded = 0
    valid = 0
    corrupted = 0
    if img_dir.is_dir():
        for p in img_dir.glob("*.jpg"):
            downloaded += 1
            try:
                from PIL import Image

                with Image.open(p) as im:
                    im.verify()
                valid += 1
            except Exception:  # noqa: BLE001
                corrupted += 1

    matched = sum(
        1
        for r in matches.values()
        if r.get("match_status") in {"matched", "matched_no_poster"}
    )
    with_poster_path = sum(1 for r in matches.values() if r.get("poster_path"))
    n_meta = len(meta)
    nonempty = sum(1 for v in captions.values() if str(v).strip())
    empty = sum(1 for v in captions.values() if not str(v).strip()) if captions else None
    cap_lens = [len(str(v)) for v in captions.values() if str(v).strip()]
    dup = 0
    if captions:
        c = Counter(str(v) for v in captions.values() if str(v).strip())
        dup = sum(1 for _k, n in c.items() if n > 1)

    # sequence items ⊆ meta
    for split_name in ("train", "val", "test"):
        split = dataset.get(split_name) or {}
        for _u, items in split.items():
            for iid in items:
                if iid not in meta:
                    errors.append(f"{split_name} item {iid} missing from meta")
                    break

    report: dict[str, Any] = {
        "path": str(path),
        "ok": not errors,
        "errors": errors,
        "num_users": len(dataset.get("umap") or {}),
        "num_items": len(dataset.get("smap") or {}),
        "meta_size": n_meta,
        "has_meta_img_des": "meta_img_des" in dataset,
        "interactions": {
            "num_users": len(dataset.get("umap") or {}),
            "num_items": len(dataset.get("smap") or {}),
        },
        "tmdb": {
            "matched_items": matched,
            "unmatched_items": max(0, len(matches) - matched),
            "items_with_poster_path": with_poster_path,
            "match_rate": (matched / len(matches)) if matches else None,
            "cache_size": len(matches),
        },
        "images": {
            "downloaded_posters": downloaded,
            "valid_posters": valid,
            "missing_posters": max(0, n_meta - downloaded),
            "corrupted_posters": corrupted,
            "poster_coverage": (downloaded / n_meta) if n_meta else None,
        },
        "captions": {
            "nonempty_captions": nonempty,
            "empty_captions": empty,
            "caption_coverage": (nonempty / n_meta) if n_meta and captions else None,
            "average_caption_length": (sum(cap_lens) / len(cap_lens)) if cap_lens else None,
            "duplicate_captions": dup,
        },
    }

    out_json = root / "validation_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md = root / "validation_report.md"
    lines = [
        "# MLLM4Rec dataset validation",
        "",
        f"- path: `{path}`",
        f"- ok: **{report['ok']}**",
        f"- users: {report['num_users']}",
        f"- items: {report['num_items']}",
        f"- meta: {report['meta_size']}",
        f"- meta_img_des: {report['has_meta_img_des']}",
        "",
        "## TMDb",
        f"- matched: {matched} / cache {len(matches)}",
        f"- with poster_path: {with_poster_path}",
        "",
        "## Images",
        f"- downloaded: {downloaded} (valid={valid}, corrupted={corrupted})",
        f"- missing vs meta: {max(0, n_meta - downloaded)}",
        "",
        "## Captions",
        f"- nonempty: {nonempty}",
        f"- empty: {empty}",
        "",
    ]
    if errors:
        lines.append("## Errors")
        lines.extend(f"- {e}" for e in errors)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
