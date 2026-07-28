"""DatasetBuilder / DatasetBundle unit tests."""

from __future__ import annotations

from llm4rec.components.dataset.base import DatasetBundle
from llm4rec.components.dataset.processor import DatasetProcessor
from llm4rec.components.dataset.schema import InteractionSchema
from llm4rec.core.schemas import Interaction


def test_dataset_bundle_extras():
    bundle = DatasetBundle(
        name="toy",
        interactions=[],
        users=["u1"],
        items=["i1", "i2"],
    )
    enriched = bundle.with_extra("semantic_ids", {"i1": [0, 1, 2]})
    assert enriched.semantic_ids["i1"] == [0, 1, 2]
    assert "semantic_ids" in enriched.summary()["extra_keys"]


def test_interaction_schema_dict():
    schema = InteractionSchema(required_extras=("images",))
    d = schema.to_dict()
    assert d["user_field"] == "user_id"
    assert "images" in d["required_extras"]


def test_processor_sequences_and_filter():
    rows = [
        Interaction(user_id="u1", item_id="a", rating=5.0, timestamp=1),
        Interaction(user_id="u1", item_id="b", rating=4.0, timestamp=2),
        Interaction(user_id="u1", item_id="c", rating=5.0, timestamp=3),
        Interaction(user_id="u2", item_id="a", rating=5.0, timestamp=1),
    ]
    seqs = DatasetProcessor.build_sequences(rows)
    assert seqs["u1"] == ["a", "b", "c"]
    filtered = DatasetProcessor.filter_min_interactions(rows, min_user=3)
    assert all(r.user_id == "u1" for r in filtered)
    users, items = DatasetProcessor.unique_ids(filtered)
    assert users == ["u1"]
    assert set(items) == {"a", "b", "c"}
