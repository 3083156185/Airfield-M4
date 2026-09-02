import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    sys.modules.pop("tools.ov.build_tips_text_cache", None)
    return importlib.import_module("tools.ov.build_tips_text_cache")


def test_build_text_cache_saves_raw_tips_features(tmp_path, monkeypatch):
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

    class FakeTokenizer:
        def tokenize(self, texts, max_len=64):
            assert texts == ["a photo of a cat", "a photo of a dog"]
            assert max_len == 64
            ids = np.array([[1, 2, 0], [3, 4, 5]], dtype=np.int32)
            paddings = np.array([[0, 0, 1], [0, 0, 0]], dtype=np.int32)
            return ids, paddings

    monkeypatch.setattr(module, "resolve_tokenizer", lambda _: FakeTokenizer())

    class FakeEncoder(torch.nn.Module):
        def forward(self, ids, paddings):
            return torch.stack(
                [ids.float().sum(dim=1), paddings.float().sum(dim=1)],
                dim=1,
            )

    model_path = tmp_path / "fake.ts"
    torch.jit.script(FakeEncoder()).save(str(model_path))

    out_path = tmp_path / "cache.pth"
    module.build_text_cache(
        ann_path=ann_path,
        out_path=out_path,
        prompt_template="a photo of a {}",
        model_path=model_path,
        tokenizer_path=tmp_path / "dummy.model",
        device="cpu",
    )

    saved = torch.load(out_path)
    assert set(saved.keys()) == {"text_feats", "prompts", "categories", "prompt_template"}
    assert saved["text_feats"].shape == (2, 2)
    expected = torch.nn.functional.normalize(torch.tensor([[3.0, 1.0], [12.0, 0.0]]), p=2, dim=-1)
    assert torch.allclose(saved["text_feats"], expected)
    assert saved["prompts"] == ["a photo of a cat", "a photo of a dog"]
    assert saved["categories"] == ["cat", "dog"]
    assert saved["prompt_template"] == "a photo of a {}"
