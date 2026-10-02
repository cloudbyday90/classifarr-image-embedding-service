# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generated artifacts reject drift, partial publication and competing writers."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from image_embedder.ir_cache import IRCache, contract_key
from image_embedder.model_catalog import MODEL_CATALOG
from image_embedder.openvino_models import ir_contract


def export_pair(xml):
    xml.write_text("<model/>", encoding="utf-8")
    xml.with_suffix(".bin").write_bytes(b"weights")


@pytest.fixture
def contract(monkeypatch):
    monkeypatch.setattr(
        "image_embedder.openvino_models.version", lambda name: "test-" + name
    )
    return ir_contract(MODEL_CATALOG["ViT-B-16"])


@pytest.mark.parametrize(
    "mutation",
    [
        "revision",
        "asset",
        "shape",
        "runtime",
        "precision",
        "preprocessing",
        "inference",
    ],
)
def test_every_generation_axis_changes_cache_identity(contract, mutation):
    modified = json.loads(json.dumps(contract))
    if mutation == "revision":
        modified["source"]["revision"] = "1" * 40
    elif mutation == "asset":
        modified["source"]["assets"]["config.json"] = "1" * 64
    elif mutation == "shape":
        modified["shape"]["image_size"] = 336
    elif mutation == "runtime":
        modified["runtime"]["openvino"] = "new-version"
    elif mutation == "precision":
        modified["export"]["compress_to_fp16"] = True
    elif mutation == "preprocessing":
        modified["preprocessing"]["policy_version"] += 1
    else:
        modified["inference"]["execution_mode"] = "PERFORMANCE"
    assert contract_key(modified) != contract_key(contract)
    assert contract_key(dict(reversed(list(contract.items())))) == contract_key(
        contract
    )


@pytest.mark.parametrize(
    "damage",
    [
        "xml",
        "bin",
        "missing_bin",
        "manifest",
        "oversized_manifest",
        "deep_manifest",
        "manifest_fifo",
        "contract",
        "symlink",
    ],
)
def test_corrupt_or_partial_entries_rebuild_before_reuse(tmp_path, contract, damage):
    cache = IRCache(tmp_path)
    xml = cache.get_or_create(contract, export_pair)
    if damage == "xml":
        xml.write_text("tampered")
    elif damage == "bin":
        xml.with_suffix(".bin").write_bytes(b"")
    elif damage == "missing_bin":
        xml.with_suffix(".bin").unlink()
    elif damage == "manifest":
        (xml.parent / "manifest.json").write_text("broken")
    elif damage == "oversized_manifest":
        (xml.parent / "manifest.json").write_bytes(b"x" * 65537)
    elif damage == "deep_manifest":
        (xml.parent / "manifest.json").write_text("[" * 2000 + "0" + "]" * 2000)
    elif damage == "manifest_fifo":
        import os

        if not hasattr(os, "mkfifo"):
            pytest.skip("named pipes unavailable on this host")
        manifest = xml.parent / "manifest.json"
        manifest.unlink()
        os.mkfifo(manifest)
    elif damage == "contract":
        manifest = json.loads((xml.parent / "manifest.json").read_text())
        manifest["contract"]["schema_version"] = 99
        (xml.parent / "manifest.json").write_text(json.dumps(manifest))
    else:
        # A manifest may not attest a linked generated weight file.
        weights = xml.with_suffix(".bin")
        target = tmp_path / "outside.bin"
        target.write_bytes(weights.read_bytes())
        weights.unlink()
        try:
            weights.symlink_to(target)
        except OSError:
            pytest.skip("symlinks unavailable on this host")
    exports = []

    def rebuild(path):
        exports.append(path)
        export_pair(path)

    assert cache.get_or_create(contract, rebuild) == xml
    assert len(exports) == 1 and xml.with_suffix(".bin").read_bytes() == b"weights"
    assert (
        cache.get_or_create(contract, lambda _p: pytest.fail("valid pair re-exported"))
        == xml
    )


def test_failed_export_cannot_publish_half_a_pair(tmp_path, contract):
    cache = IRCache(tmp_path)

    def incomplete(xml):
        xml.write_text("<model/>")

    with pytest.raises(ValueError, match="nonempty regular"):
        cache.get_or_create(contract, incomplete)
    assert not (tmp_path / contract_key(contract)).exists()
    assert not list(tmp_path.glob(".build-*"))
    assert cache.get_or_create(contract, export_pair).exists()


def test_concurrent_cache_owners_export_once(tmp_path, contract):
    exports = []

    def build(path):
        exports.append(path)
        export_pair(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(
            pool.map(
                lambda _: IRCache(tmp_path).get_or_create(contract, build), range(24)
            )
        )
    assert len(exports) == 1 and len(set(paths)) == 1


def test_process_lock_has_a_finite_timeout(tmp_path, contract):
    with FileLock(tmp_path / f"{contract_key(contract)}.lock"):
        with pytest.raises(Timeout):
            IRCache(tmp_path, lock_timeout=0).get_or_create(contract, export_pair)


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_invalid_lock_timeouts_are_rejected(tmp_path, timeout):
    with pytest.raises(ValueError):
        IRCache(tmp_path, lock_timeout=timeout)


def _process_export(args):
    root, contract = args

    def build(xml):
        with (Path(root) / "exports.log").open("a", encoding="utf-8") as stream:
            stream.write("export\n")
        export_pair(xml)

    return str(IRCache(Path(root)).get_or_create(contract, build))


def test_independent_processes_publish_one_complete_pair(tmp_path, contract):
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        paths = list(pool.map(_process_export, [(str(tmp_path), contract)] * 12))
    assert len(set(paths)) == 1
    assert (tmp_path / "exports.log").read_text(encoding="utf-8") == "export\n"
    assert Path(paths[0]).read_text(encoding="utf-8") == "<model/>"
    assert Path(paths[0]).with_suffix(".bin").read_bytes() == b"weights"


def test_unverified_spec_cannot_silently_choose_weights():
    with pytest.raises(ValueError, match="verified weights"):
        replace(MODEL_CATALOG["ViT-B-16"], assets=()).weights_file
