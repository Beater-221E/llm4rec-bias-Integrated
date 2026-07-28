"""Rebuild RecommendationExample prompts without parsing prompt strings."""

from __future__ import annotations

import copy
from typing import Any, Sequence

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError
from llm4rec_bias_Integrated.core.schemas import RecommendationExample
from llm4rec_bias_Integrated.prompts.candidate_choice import LETTERS, build_candidate_choice_messages


def _history_titles(ex: RecommendationExample) -> list[str]:
    titles = ex.features.get("history_titles")
    if titles is None:
        raise ConfigurationError(
            f"example {ex.example_id} missing features.history_titles "
            "(rebuild requires structured titles; do not parse prompts)"
        )
    return list(titles)


def _candidate_titles(ex: RecommendationExample) -> list[str]:
    titles = ex.features.get("candidate_titles")
    if titles is None:
        raise ConfigurationError(
            f"example {ex.example_id} missing features.candidate_titles"
        )
    return list(titles)


def _pop_quantiles(ex: RecommendationExample) -> list[float]:
    quants = ex.features.get("pop_quantiles")
    if quants is None:
        raise ConfigurationError(
            f"example {ex.example_id} missing features.pop_quantiles"
        )
    return [float(q) for q in quants]


def _genres(ex: RecommendationExample) -> list[tuple[str, ...]] | None:
    raw = ex.features.get("candidate_genres")
    if raw is None:
        return None
    return [tuple(g) for g in raw]


def _years(ex: RecommendationExample) -> list[int | None] | None:
    raw = ex.features.get("candidate_years")
    if raw is None:
        return None
    return list(raw)


def rebuild_example(
    ex: RecommendationExample,
    *,
    history_titles: Sequence[str] | None = None,
    history_item_ids: Sequence[str] | None = None,
    candidate_ids: Sequence[str] | None = None,
    candidate_titles: Sequence[str] | None = None,
    pop_quantiles: Sequence[float] | None = None,
    target_index: int | None = None,
    framing: str | None = None,
    candidate_genres: Sequence[Sequence[str]] | None = None,
    candidate_years: Sequence[int | None] | None = None,
    extra_features: dict[str, Any] | None = None,
    example_id_suffix: str = "",
) -> RecommendationExample:
    """Clone ``ex`` with rewritten structured fields and rebuilt prompt messages."""
    hist_titles = list(history_titles) if history_titles is not None else _history_titles(ex)
    cand_titles = (
        list(candidate_titles) if candidate_titles is not None else _candidate_titles(ex)
    )
    quants = (
        [float(q) for q in pop_quantiles]
        if pop_quantiles is not None
        else _pop_quantiles(ex)
    )
    if len(cand_titles) != len(quants):
        raise ConfigurationError("candidate_titles and pop_quantiles length mismatch")

    if target_index is None:
        target_index = ex.target_index
    if target_index is None:
        raise ConfigurationError(f"example {ex.example_id} missing target_index")
    if not (0 <= int(target_index) < len(cand_titles)):
        raise ConfigurationError(
            f"target_index {target_index} out of range for {len(cand_titles)} candidates"
        )

    frame = framing if framing is not None else str(ex.features.get("framing") or "neutral")
    genres = (
        [tuple(g) for g in candidate_genres]
        if candidate_genres is not None
        else _genres(ex)
    )
    years = (
        list(candidate_years) if candidate_years is not None else _years(ex)
    )

    messages = build_candidate_choice_messages(
        hist_titles,
        cand_titles,
        quants,
        frame,
        candidate_genres=genres,
        candidate_years=years,
    )

    feats = copy.deepcopy(ex.features)
    feats["history_titles"] = hist_titles
    feats["candidate_titles"] = cand_titles
    feats["pop_quantiles"] = quants
    feats["framing"] = frame
    feats["candidate_positions"] = list(range(len(cand_titles)))
    if genres is not None:
        feats["candidate_genres"] = [list(g) for g in genres]
    if years is not None:
        feats["candidate_years"] = list(years)
    if extra_features:
        feats.update(extra_features)

    cands = list(candidate_ids) if candidate_ids is not None else list(ex.candidates or [])
    if candidate_ids is not None and len(cands) != len(cand_titles):
        raise ConfigurationError("candidate_ids and candidate_titles length mismatch")

    new_id = ex.example_id + (example_id_suffix or "")
    return RecommendationExample(
        example_id=new_id,
        user_id=ex.user_id,
        history_item_ids=(
            list(history_item_ids)
            if history_item_ids is not None
            else list(ex.history_item_ids)
        ),
        target_item_id=ex.target_item_id,
        candidates=cands if cands else None,
        prompt_messages=messages,
        target_text=LETTERS[int(target_index)],
        target_index=int(target_index),
        semantic_id=ex.semantic_id,
        features=feats,
    )


def place_target_at_slot(
    ex: RecommendationExample,
    slot: int,
) -> RecommendationExample:
    """Re-place the same target + negatives so the target sits at ``slot``."""
    if ex.candidates is None or ex.target_index is None:
        raise ConfigurationError("place_target_at_slot requires candidates and target_index")
    titles = _candidate_titles(ex)
    quants = _pop_quantiles(ex)
    ids = list(ex.candidates)
    t = int(ex.target_index)
    n = len(titles)
    if not (0 <= slot < n):
        raise ConfigurationError(f"slot {slot} out of range 0..{n - 1}")

    tgt_title, tgt_q, tgt_id = titles[t], quants[t], ids[t]
    neg_titles = [x for i, x in enumerate(titles) if i != t]
    neg_quants = [x for i, x in enumerate(quants) if i != t]
    neg_ids = [x for i, x in enumerate(ids) if i != t]

    genres = _genres(ex)
    years = _years(ex)
    neg_genres = None
    neg_years = None
    tgt_genres = ()
    tgt_year = None
    if genres is not None:
        tgt_genres = genres[t]
        neg_genres = [g for i, g in enumerate(genres) if i != t]
    if years is not None:
        tgt_year = years[t]
        neg_years = [y for i, y in enumerate(years) if i != t]

    new_titles = neg_titles[:slot] + [tgt_title] + neg_titles[slot:]
    new_quants = neg_quants[:slot] + [tgt_q] + neg_quants[slot:]
    new_ids = neg_ids[:slot] + [tgt_id] + neg_ids[slot:]
    new_genres = None
    new_years = None
    if neg_genres is not None:
        new_genres = neg_genres[:slot] + [tgt_genres] + neg_genres[slot:]
    if neg_years is not None:
        new_years = neg_years[:slot] + [tgt_year] + neg_years[slot:]

    return rebuild_example(
        ex,
        candidate_ids=new_ids,
        candidate_titles=new_titles,
        pop_quantiles=new_quants,
        target_index=slot,
        candidate_genres=new_genres,
        candidate_years=new_years,
        extra_features={"probe_target_slot": slot},
        example_id_suffix=f":pos{slot}",
    )
