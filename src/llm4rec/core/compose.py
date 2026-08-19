"""分层配置组合器。

一个实验 = ``configs/exp/<name>.yaml``，顶部用 ``defaults:`` 列出要先加载的层：

    defaults:
      - base
      - data/amazon23
      - model/qwen2.5-0.5b-instruct
      - sid/rqvae
      - bias/default

按顺序深度合并，实验文件自己的键最后覆盖，再叠加 CLI 的 ``a.b.c=value``。
``defaults`` 可以嵌套（被引用的层自己也能有 ``defaults``），环路会被检测并报错。

设计意图：训练的同学只需要动 ``configs/exp/*.yaml`` 和 ``run.sh``，
不必再去一个 500 行的巨型 config 里翻三层嵌套。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from llm4rec.core.exceptions import ConfigurationError
from llm4rec.core.paths import configs_dir

_ROUTES = ("minionerec", "recr1", "dpo4rec")

# 每个 route 必须给出的 stage 集合，validate 时校验
_ROUTE_STAGES: dict[str, set[str]] = {
    "minionerec": {"sft", "rl", "eval"},
    # recr1 / dpo4rec 原文都没有 SFT，我们统一加，好让三条路线有同一种基线
    "recr1": {"sft", "rl", "eval"},
    "dpo4rec": {"train_reranker", "sft", "dpo", "eval"},
}

# 哪个 stage 用 train 配置里的哪个块
STAGE_TRAIN_KEY: dict[str, str] = {"sft": "sft", "rl": "rl", "dpo": "dpo"}


def _resolve_layer_path(name: str, root: Path) -> Path:
    """``data/amazon23`` → ``configs/data/amazon23.yaml``。"""
    text = name.strip()
    if text.endswith((".yaml", ".yml")):
        candidates = [root / text]
    else:
        candidates = [root / f"{text}.yaml", root / f"{text}.yml"]
    for path in candidates:
        if path.is_file():
            return path
    raise ConfigurationError(
        f"配置层 '{name}' 不存在，找过：{[str(p) for p in candidates]}"
    )


def _load_layer(path: Path, root: Path, seen: list[Path]) -> dict[str, Any]:
    """加载一个 YAML 层，先递归展开它自己的 ``defaults``。"""
    resolved = path.resolve()
    if resolved in seen:
        chain = " → ".join(p.name for p in [*seen, resolved])
        raise ConfigurationError(f"配置 defaults 存在环路：{chain}")

    raw = OmegaConf.load(path)
    node = OmegaConf.to_container(raw, resolve=False)
    if not isinstance(node, dict):
        raise ConfigurationError(f"{path} 顶层必须是 mapping")

    defaults = node.pop("defaults", None) or []
    if not isinstance(defaults, list):
        raise ConfigurationError(f"{path} 的 defaults 必须是列表")

    merged: dict[str, Any] = {}
    for entry in defaults:
        if not isinstance(entry, str):
            raise ConfigurationError(f"{path} 的 defaults 只支持字符串条目，得到 {entry!r}")
        child = _load_layer(
            _resolve_layer_path(entry, root), root, [*seen, resolved]
        )
        merged = _deep_merge(merged, child)

    return _deep_merge(merged, node)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """深度合并；``overlay`` 的标量/列表整体覆盖 ``base``。"""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_overrides(tokens: list[str]) -> dict[str, Any]:
    """把 ``["train.sft.learning_rate=2e-5", ...]`` 解析成 dotted-key → 值。"""
    out: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ConfigurationError(f"非法 override '{token}'，格式应为 key=value")
        key, raw = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigurationError(f"override '{token}' 的 key 为空")
        out[key] = _parse_scalar(raw.strip())
    return out


def _parse_scalar(raw: str) -> Any:
    if raw == "" or raw.lower() in {"null", "none", "~"}:
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.startswith(("[", "{")):
        try:
            return OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"无法解析结构化 override: {raw}") from exc
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def compose(
    experiment: str,
    overrides: list[str] | None = None,
    *,
    config_root: Path | None = None,
) -> DictConfig:
    """组合出一份完整的运行时配置。

    ``experiment`` 可以是 ``minionerec_qwen05b_amazon``
    （找 ``configs/exp/minionerec_qwen05b_amazon.yaml``）、
    ``exp/minionerec_qwen05b_amazon``，或者一个直接的 yaml 路径。
    """
    root = config_root or configs_dir()

    text = experiment.strip()
    path: Path | None = None
    for candidate in (text, f"exp/{text}"):
        try:
            path = _resolve_layer_path(candidate, root)
            break
        except ConfigurationError:
            continue
    if path is None:
        available = sorted(p.stem for p in (root / "exp").glob("*.yaml"))
        raise ConfigurationError(
            f"找不到实验 '{experiment}'。可用：{available}"
        )

    merged = _load_layer(path, root, [])

    cfg = OmegaConf.create(merged)
    for key, value in parse_overrides(overrides or []).items():
        OmegaConf.update(cfg, key, value, merge=True)

    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def load_layer(name: str, *, config_root: Path | None = None) -> dict[str, Any]:
    """单独加载某一层（不做实验组合）。

    给 DeepSpeed 这类"按名字挑一份配置"的场景用：
    ``load_layer("deepspeed/zero2")`` → ``configs/deepspeed/zero2.yaml`` 的内容。
    """
    root = config_root or configs_dir()
    return _load_layer(_resolve_layer_path(name, root), root, [])


def to_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise ConfigurationError("解析后的配置必须是 mapping")
    return container


def validate(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """在真正加载模型/数据之前把配置错误挡下来。"""
    data = to_dict(cfg)

    # Apply reproduction / integrated mode defaults before validation.
    from llm4rec.core.modes import apply_mode_defaults, get_mode, verify_minionerec_reproduction

    data = apply_mode_defaults(data)
    mode = get_mode(data)

    for key in ("experiment", "data", "model", "stages", "paths", "wandb", "bias"):
        if key not in data:
            raise ConfigurationError(f"配置缺少顶层键 '{key}'")

    exp = data["experiment"]
    if not isinstance(exp, dict) or not exp.get("name"):
        raise ConfigurationError("experiment.name 必填")
    route = str(exp.get("route") or "")
    if route not in _ROUTES:
        raise ConfigurationError(
            f"experiment.route 必须是 {_ROUTES} 之一，得到 {route!r}"
        )
    data["mode"] = mode
    exp["mode"] = mode

    stages = data["stages"]
    if not isinstance(stages, list) or not stages:
        raise ConfigurationError("stages 必须是非空列表")
    # Reproduction Rec-R1 may omit SFT (official has no SFT stage).
    allowed = set(_ROUTE_STAGES[route])
    if route == "recr1" and mode == "reproduction":
        allowed = {"rl", "eval", "sft"}  # sft optional
    unknown = [s for s in stages if s not in allowed]
    if unknown:
        raise ConfigurationError(
            f"route '{route}' 不支持这些 stage: {unknown}（可用：{sorted(allowed)}）"
        )
    if route == "recr1" and mode == "reproduction" and "sft" in stages:
        # Allowed but documented as an integrated deviation.
        pass

    model = data["model"]
    if not isinstance(model, dict) or not model.get("checkpoint"):
        raise ConfigurationError("model.checkpoint 必填")
    # LoRA 已从本框架移除，撞见残留配置直接报错而不是静默忽略
    if data.get("peft") or model.get("use_lora"):
        raise ConfigurationError(
            "本框架已移除 LoRA：MiniOneRec 官方 SFT 是全参微调，"
            "且 LoRA 会低估 RL 对表征的改动。请删掉 peft / model.use_lora 配置。"
        )

    seed = data.get("seed")
    if not isinstance(seed, int):
        raise ConfigurationError(f"seed 必须是 int，得到 {type(seed).__name__}")

    bias = data["bias"]
    if not isinstance(bias, dict):
        raise ConfigurationError("bias 必须是 mapping")
    online = bias.get("online_stages") or []
    if "sft" in online:
        raise ConfigurationError(
            "bias.online_stages 不应包含 sft：我们的假设是 RL 放大 bias，"
            "SFT 只在结束时评一次基线。"
        )

    # 每个要跑的训练 stage 都得有对应的 train.<stage> 配置块
    train = data.get("train") or {}
    for stage in stages:
        key = STAGE_TRAIN_KEY.get(stage)
        if key and key not in train:
            raise ConfigurationError(
                f"stages 里有 '{stage}'，但配置缺少 train.{key} 块"
            )

    if mode == "reproduction" and route == "minionerec":
        verify_minionerec_reproduction(data)

    return data


def stage_eval_steps(cfg: dict[str, Any], stage: str) -> tuple[int, int]:
    """返回 ``(准确率 eval 频率, bias 在线评测频率)``，单位是 optimizer step。

    RL 是按 step 跑的（不是按 epoch），所以评测频率也在 ``train.<stage>``
    里按 step 配。``bias_eval_steps`` 留 null 就跟随 ``eval_steps``。
    """
    key = STAGE_TRAIN_KEY.get(stage, stage)
    block = (cfg.get("train") or {}).get(key) or {}
    eval_steps = int(block.get("eval_steps") or 0) or 50
    bias_steps = block.get("bias_eval_steps")
    if bias_steps in (None, 0, "null"):
        bias_steps = eval_steps
    return eval_steps, int(bias_steps)
