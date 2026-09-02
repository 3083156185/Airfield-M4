import importlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    sys.modules.pop("tools.ov.build_mobileclip_text_cache", None)
    return importlib.import_module("tools.ov.build_mobileclip_text_cache")


def test_build_text_cache_saves_mobileclip_features_and_prompts(tmp_path, monkeypatch):
    module = _load_module()

    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 2, "name": "dog"},
                    {"id": 1, "name": "cat"},
                ]
            }
        )
    )

    monkeypatch.setattr(
        module,
        "resolve_tokenizer",
        lambda **kwargs: (lambda texts: torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.int64)),
    )

    class FakeEncoder(torch.nn.Module):
        def forward(self, tokens):
            return torch.stack([tokens.float().sum(dim=1), tokens.float().max(dim=1).values], dim=1)

    model_path = tmp_path / "fake.ts"
    torch.jit.script(FakeEncoder()).save(str(model_path))

    out_path = tmp_path / "cache.pth"
    module.build_text_cache(
        ann_path=ann_path,
        out_path=out_path,
        prompt_template="a photo of a {}",
        model_path=model_path,
        device="cpu",
        tokenizer_path="local-tokenizer",
        tokenizer_local_files_only=True,
    )

    saved = torch.load(out_path)
    assert set(saved.keys()) == {"text_feats", "prompts", "categories", "prompt_template"}
    assert saved["text_feats"].shape == (2, 2)
    assert torch.equal(saved["text_feats"], torch.tensor([[3.0, 2.0], [12.0, 5.0]]))
    assert saved["prompts"] == ["a photo of a cat", "a photo of a dog"]
    assert saved["categories"] == ["cat", "dog"]
    assert saved["prompt_template"] == "a photo of a {}"


def test_resolve_tokenizer_loads_hf_tokenizer_from_configured_local_path(monkeypatch):
    module = _load_module()
    sys.modules.pop("clip", None)

    calls = []

    class FakeHFTokenizer:
        @classmethod
        def from_pretrained(cls, tokenizer_path, **kwargs):
            calls.append((tokenizer_path, kwargs))
            return cls()

        def __call__(self, texts, **kwargs):
            assert texts == ["a photo of a cat"]
            assert kwargs == {
                "padding": "max_length",
                "truncation": True,
                "max_length": 77,
                "return_tensors": "pt",
            }
            return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64)}

    class FakeTransformers:
        CLIPTokenizerFast = FakeHFTokenizer

    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)

    tokenizer = module.resolve_tokenizer(
        tokenizer_path="pretrained/clip-vit-base-patch32",
        tokenizer_local_files_only=True,
    )

    assert torch.equal(tokenizer(["a photo of a cat"]), torch.tensor([[1, 2, 3]], dtype=torch.int64))
    assert calls == [
        (
            "pretrained/clip-vit-base-patch32",
            {"local_files_only": True},
        )
    ]
