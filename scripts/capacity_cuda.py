# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""CUDA observations and synchronized phase boundaries for a dedicated probe."""


class CudaReader:
    def __init__(self, cuda) -> None:
        if not cuda.is_available():
            raise ValueError("CUDA capacity calibration requires an available GPU")
        self.cuda = cuda
        cuda.init()
        self.device = cuda.current_device()

    def begin_phase(self) -> None:
        self.cuda.synchronize(self.device)
        self.cuda.reset_peak_memory_stats(self.device)

    def end_phase(self) -> None:
        self.cuda.synchronize(self.device)

    def __call__(self) -> dict:
        free, total = self.cuda.mem_get_info(self.device)
        return {
            "device_index": self.device,
            "device_name": self.cuda.get_device_name(self.device),
            "allocator_backend": self.cuda.get_allocator_backend(),
            "allocated_bytes": self.cuda.memory_allocated(self.device),
            "reserved_bytes": self.cuda.memory_reserved(self.device),
            "phase_peak_allocated_bytes": self.cuda.max_memory_allocated(self.device),
            "phase_peak_reserved_bytes": self.cuda.max_memory_reserved(self.device),
            "device_free_bytes": free,
            "device_total_bytes": total,
        }
