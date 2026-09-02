import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.misc.modality_utils import normalize_tensor_minmax_per_sample


def test_normalize_tensor_minmax_per_sample_supports_non_contiguous_tensor():
    base = torch.tensor(
        [
                [
                    [-1.0, -0.5, 0.0, 0.2],
                    [-0.2, 0.1, 0.4, 0.8],
                    [0.0, 0.3, 0.6, 1.0],
                    [0.2, 0.5, 0.9, 0.9],
                ]
            ],
            dtype=torch.float32,
    )
    cropped = base[:, 1:4, 1:4]
    assert cropped.is_contiguous() is False

    normalized = normalize_tensor_minmax_per_sample(cropped)

    assert normalized.shape == cropped.shape
    assert torch.allclose(normalized, cropped.clamp(0.0, 1.0))
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_tensor_minmax_per_sample_clamps_unit_range_without_rescaling():
    tensor = torch.tensor([[[-0.2, 0.2], [0.7, 1.0]]], dtype=torch.float32)

    normalized = normalize_tensor_minmax_per_sample(tensor)

    expected = torch.tensor([[[0.0, 0.2], [0.7, 1.0]]], dtype=torch.float32)
    assert torch.allclose(normalized, expected)


def test_normalize_tensor_minmax_per_sample_scales_values_over_one_by_255():
    tensor = torch.tensor([[[0.0, 64.0], [128.0, 255.0]]], dtype=torch.float32)

    normalized = normalize_tensor_minmax_per_sample(tensor)

    expected = tensor / 255.0
    assert torch.allclose(normalized, expected)
