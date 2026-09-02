from pathlib import Path

import pytest

from tools.inference.multimodal.torch_inf import iter_multimodal_inputs


def test_iter_multimodal_inputs_pairs_rgb_directory_by_stem(tmp_path):
    rgb_dir = tmp_path / "rgb"
    npy_dir = tmp_path / "npy"
    rgb_dir.mkdir()
    npy_dir.mkdir()

    (rgb_dir / "b.png").touch()
    (rgb_dir / "a.jpg").touch()
    (rgb_dir / "ignore.txt").touch()
    (npy_dir / "a.npy").touch()
    (npy_dir / "b.npz").touch()

    pairs = iter_multimodal_inputs(rgb_dir, npy_dir)

    assert pairs == [
        (rgb_dir / "a.jpg", npy_dir / "a.npy"),
        (rgb_dir / "b.png", npy_dir / "b.npz"),
    ]


def test_iter_multimodal_inputs_resolves_single_rgb_from_npy_directory(tmp_path):
    rgb_path = tmp_path / "frame.jpeg"
    npy_dir = tmp_path / "npy"
    rgb_path.touch()
    npy_dir.mkdir()
    (npy_dir / "frame.npz").touch()

    assert iter_multimodal_inputs(rgb_path, npy_dir) == [(rgb_path, npy_dir / "frame.npz")]


def test_iter_multimodal_inputs_allows_explicit_single_file_pair(tmp_path):
    rgb_path = tmp_path / "rgb.bmp"
    npy_path = tmp_path / "thermal.npy"
    rgb_path.touch()
    npy_path.touch()

    assert iter_multimodal_inputs(rgb_path, npy_path) == [(rgb_path, npy_path)]


def test_iter_multimodal_inputs_rejects_rgb_directory_with_single_npy_file(tmp_path):
    rgb_dir = tmp_path / "rgb"
    npy_path = tmp_path / "thermal.npy"
    rgb_dir.mkdir()
    (rgb_dir / "a.jpg").touch()
    npy_path.touch()

    with pytest.raises(ValueError, match="directory"):
        iter_multimodal_inputs(rgb_dir, npy_path)


def test_iter_multimodal_inputs_raises_for_missing_modality_file(tmp_path):
    rgb_path = tmp_path / "frame.jpg"
    npy_dir = tmp_path / "npy"
    rgb_path.touch()
    npy_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="frame.npy"):
        iter_multimodal_inputs(rgb_path, npy_dir)
