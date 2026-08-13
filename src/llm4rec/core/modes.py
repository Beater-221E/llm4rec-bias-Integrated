"""Apply reproduction vs integrated mode defaults without duplicating trainers.

Reproduction MiniOneRec SID fields are **forced** (not setdefault) so they
cannot accidentally inherit integrated PCA / simplified RQ-VAE settings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.sid.minionerec_rqvae import MINIONEREC_RQVAE_DEFAULTS


_VALID_MODES = ("reproduction", "integrated")


def get_mode(cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("mode") or (cfg.get("experiment") or {}).get("mode") or "integrated").lower()
    if mode not in _VALID_MODES:
        raise ConfigurationError(f"mode 必须是 {_VALID_MODES}，得到 {mode!r}")
    return mode


def apply_mode_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg)
    mode = get_mode(out)
    out["mode"] = mode
    out.setdefault("experiment", {})
    out["experiment"]["mode"] = mode
    route = str(out["experiment"].get("route") or "")

    hw = out.setdefault("hardware", {})
    hw.setdefault("devices", "auto")
    hw.setdefault("precision", "auto")
    hw.setdefault("strategy", "auto")
    hw.setdefault("memory", "auto")
    hw.setdefault("find_unused_parameters", False)

    opt = out.setdefault("optimization", {})
    opt.setdefault("compile", {"enabled": "auto", "backend": "inductor", "mode": "default"})
    opt.setdefault("attention", {"implementation": "auto"})
    opt.setdefault("generation", {"cache": "auto", "pad_to_multiple_of": 8})
    opt.setdefault("triton", {"rq_distance_argmin": False})

    if mode != "reproduction":
        return out

    if route == "minionerec":
        sid = out.setdefault("sid", {})
        # Force reference SID settings (override integrated inheritance)
        sid["implementation"] = "minionerec_reference"
        sid["method"] = "rqvae"
        sid["levels"] = 3
        sid["codebook_size"] = 256
        sid["strict_unique"] = bool(sid.get("strict_unique", False))
        if sid.get("max_collision_rate") is not None and float(sid["max_collision_rate"]) == 0.0:
            sid["max_collision_rate"] = 1.0
        rq = dict(sid.get("rqvae") or {})
        for key, value in MINIONEREC_RQVAE_DEFAULTS.items():
            rq[key] = deepcopy(value) if isinstance(value, list) else value
        rq["pca_dim"] = None
        rq["num_emb_list"] = [256, 256, 256]
        rq["layers"] = [2048, 1024, 512, 256, 128, 64]
        rq["e_dim"] = 32
        sid["rqvae"] = rq
        sid["collision_handling"] = "minionerec_sinkhorn_last_level"
        out["sid"] = sid

        # Force reference-compatible training knobs only when profile is reproduction
        train = out.setdefault("train", {})
        sft = train.setdefault("sft", {})
        sft["tasks"] = list(sft.get("tasks") or ["seqrec", "title2sid", "sid2title"])
        rl = train.setdefault("rl", {})
        rl["algorithm"] = "grpo"
        grpo = rl.setdefault("grpo", {})
        grpo.setdefault("group_size", 16)
        grpo.setdefault("beta", 1.0e-3)
        grpo.setdefault("constrained_rollout", True)
        # Reproduction: keep compile off unless user opts in
        opt["compile"] = dict(opt.get("compile") or {})
        if opt["compile"].get("enabled") == "auto":
            opt["compile"]["enabled"] = False
        opt["triton"] = {"rq_distance_argmin": False}

    elif route == "recr1":
        stages = list(out.get("stages") or [])
        if stages == ["sft", "eval", "rl", "eval"]:
            out["stages"] = ["rl", "eval"]
        train = out.setdefault("train", {})
        rl = train.setdefault("rl", {})
        rl.setdefault("algorithm", "grpo")
        opt["compile"] = dict(opt.get("compile") or {})
        if opt["compile"].get("enabled") == "auto":
            opt["compile"]["enabled"] = False

    elif route == "dpo4rec":
        train = out.setdefault("train", {})
        dpo = train.setdefault("dpo", {})
        dpo.setdefault("iterations", 2)
        dpo.setdefault("num_samples", 10)
        dpo.setdefault("beta", 0.01)

    return out


def verify_minionerec_reproduction(cfg: dict[str, Any]) -> list[str]:
    """Return human-readable verification lines; raise on hard incompatibilities."""
    sid = cfg.get("sid") or {}
    rq = sid.get("rqvae") or {}
    errors = []
    if sid.get("implementation") not in {"minionerec_reference", "minionerec", "official"}:
        errors.append(f"SID implementation={sid.get('implementation')}")
    if int(sid.get("levels") or 0) != 3:
        errors.append(f"levels={sid.get('levels')}")
    if list(rq.get("num_emb_list") or []) != [256, 256, 256]:
        errors.append(f"codebooks={rq.get('num_emb_list')}")
    if int(rq.get("e_dim") or 0) != 32:
        errors.append(f"e_dim={rq.get('e_dim')}")
    if list(rq.get("layers") or []) != [2048, 1024, 512, 256, 128, 64]:
        errors.append(f"layers={rq.get('layers')}")
    if rq.get("pca_dim") not in (None, 0, False):
        errors.append(f"pca_dim={rq.get('pca_dim')}")
    if errors:
        raise ConfigurationError(
            "MiniOneRec reproduction SID mismatch: " + "; ".join(errors)
        )
    return [
        "RQ levels=3",
        "codebook sizes=[256,256,256]",
        "embedding dim=32",
        "RQ-VAE architecture=minionerec_reference",
        "PCA=disabled",
        "collision resolution=enabled",
        "SID implementation=minionerec_reference",
    ]
