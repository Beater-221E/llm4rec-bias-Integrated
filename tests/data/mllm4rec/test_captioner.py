"""BLIP2 captioner tests (fully mocked — no model download)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from llm4rec_bias_Integrated.data.mllm4rec.blip2_captioner import (
    generate_captions_for_dataset,
    load_caption_cache,
)


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(path, format="JPEG")


def test_caption_mock_and_missing_image(tmp_path: Path) -> None:
    img_dir = tmp_path / "img"
    _write_jpg(img_dir / "1.jpg")
    dataset = {"meta": {1: "A (1995)", 2: "B (1995)"}}
    captions_path = tmp_path / "captions.jsonl"

    proc_inst = MagicMock()
    inputs = MagicMock()
    inputs.to.return_value = {"pixel_values": MagicMock()}
    proc_inst.return_value = inputs
    proc_inst.batch_decode.return_value = [" a toy story poster "]

    model = MagicMock()
    model.generate.return_value = [[1, 2, 3]]
    model.to = MagicMock(return_value=model)

    with (
        patch(
            "transformers.Blip2Processor.from_pretrained",
            return_value=proc_inst,
        ),
        patch(
            "transformers.Blip2ForConditionalGeneration.from_pretrained",
            return_value=model,
        ),
    ):
        out = generate_captions_for_dataset(
            dataset,
            img_dir=img_dir,
            model_name_or_path="Salesforce/blip2-opt-2.7b",
            device="cpu",
            dtype="float32",
            mode="original",
            resume=False,
            captions_path=captions_path,
        )

    assert out[1] == "a toy story poster"
    assert out[2] == ""
    assert set(dataset["meta"].keys()) == set(dataset["meta_img_des"].keys())
    assert model.generate.call_count >= 1
    n_calls = model.generate.call_count

    with (
        patch(
            "transformers.Blip2Processor.from_pretrained",
            return_value=proc_inst,
        ),
        patch(
            "transformers.Blip2ForConditionalGeneration.from_pretrained",
            return_value=model,
        ),
    ):
        generate_captions_for_dataset(
            dataset,
            img_dir=img_dir,
            model_name_or_path="Salesforce/blip2-opt-2.7b",
            device="cpu",
            dtype="float32",
            mode="original",
            resume=True,
            captions_path=captions_path,
        )
    assert model.generate.call_count == n_calls
    assert load_caption_cache(captions_path)[1] == "a toy story poster"


def test_exception_writes_empty_caption(tmp_path: Path) -> None:
    img_dir = tmp_path / "img"
    _write_jpg(img_dir / "1.jpg")
    dataset = {"meta": {1: "A (1995)"}}

    proc_inst = MagicMock()
    proc_inst.side_effect = RuntimeError("boom")

    model = MagicMock()
    model.to = MagicMock(return_value=model)

    with (
        patch(
            "transformers.Blip2Processor.from_pretrained",
            return_value=proc_inst,
        ),
        patch(
            "transformers.Blip2ForConditionalGeneration.from_pretrained",
            return_value=model,
        ),
    ):
        out = generate_captions_for_dataset(
            dataset,
            img_dir=img_dir,
            model_name_or_path="Salesforce/blip2-opt-2.7b",
            device="cpu",
            dtype="float32",
            mode="original",
            resume=False,
            captions_path=tmp_path / "captions.jsonl",
        )
    assert out[1] == ""
