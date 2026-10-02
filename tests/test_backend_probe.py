# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backend smoke checks reject substituted wheels and exercise offline CLIP."""

import pytest
from backend_probe import probe_backend, validate_torch_backend


@pytest.mark.parametrize("backend,cuda", [("cpu", None), ("openvino", None), ("cuda", "13.0"), ("cuda-legacy", "12.6")])
def test_expected_torch_profile(backend, cuda):
    validate_torch_backend(backend, cuda, None)


@pytest.mark.parametrize(
    "backend,cuda,hip",
    [
        ("cpu", "13.0", None),
        ("openvino", "12.8", None),
        ("cuda", None, None),
        ("cuda", "12.4", None),
        ("cuda", "12.8", None),
        ("cuda", "12.6", None),
        ("cuda-legacy", "13.0", None),
        ("cpu", None, "7.2"),
        ("cuda", "13.0", "7.2"),
    ],
)
def test_wrong_torch_profile_fails(backend, cuda, hip):
    with pytest.raises(RuntimeError, match="Wrong Torch profile"):
        validate_torch_backend(backend, cuda, hip)


def test_unknown_backend_fails():
    with pytest.raises(ValueError, match="Unknown backend"):
        validate_torch_backend("typo", None, None)


def test_offline_clip_projection_and_cache_access(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    sentinel = tmp_path / "existing-model.xml"
    sentinel.write_text("keep", encoding="utf-8")
    result = probe_backend("cpu", tmp_path)
    assert result["projection_shape"] == [1, 8]
    assert result["cuda"] is None
    assert sorted(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_required_gpu_cannot_succeed_on_cpu(tmp_path):
    with pytest.raises(RuntimeError, match="GPU execution was required"):
        probe_backend("cpu", tmp_path, require_gpu=True)
