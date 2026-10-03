# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from types import SimpleNamespace

import pytest
from capacity_cuda import CudaReader
from test_capacity_workload import workload


def test_unavailable_cuda_never_claims_cpu_as_gpu():
    with pytest.raises(ValueError, match="available GPU"):
        CudaReader(SimpleNamespace(is_available=lambda: False))


def test_cuda_phase_peaks_reset_once_and_sampling_never_synchronizes():
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        init=lambda: calls.append("init"),
        current_device=lambda: 0,
        synchronize=lambda _: calls.append("sync"),
        reset_peak_memory_stats=lambda _: calls.append("reset"),
        get_device_name=lambda _: "fixture",
        get_allocator_backend=lambda: "native",
        memory_allocated=lambda _: 10,
        memory_reserved=lambda _: 20,
        max_memory_allocated=lambda _: 30,
        max_memory_reserved=lambda _: 40,
        mem_get_info=lambda _: (50, 100),
    )
    reader = CudaReader(cuda)
    subject = workload([])
    subject.cuda = reader

    def native():
        calls.append("work")
        assert reader()["phase_peak_reserved_bytes"] == 40
        assert reader()["device_free_bytes"] == 50

    subject.measure("cuda", native)
    assert calls == ["init", "sync", "reset", "work", "sync"]
