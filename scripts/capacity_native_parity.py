# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Probe facade: validate native results even after their HTTP waiter leaves."""

import numpy as np

from image_embedder.execution_cancellation import InferenceCancelled


class NativeBatchVerifier:
    def __init__(self, embedder, fixtures, references, timings):
        self.embedder, self.references, self.timings = embedder, references, timings
        self.inputs = {
            payload: (fixture.name, variant)
            for fixture in fixtures
            for variant, payload in enumerate(fixture.payloads)
        }

    def __getattr__(self, name):
        # All other methods stay bound to the original owner; no models are copied.
        return getattr(self.embedder, name)

    def embed_batch(self, spec, target_size, items):
        trace = self.timings.current.get()
        if trace is None:
            raise AssertionError("Native representative batch lacks request context")
        try:
            results = self.embedder.embed_batch(spec, target_size, items)
            if len(results) != len(items):
                raise ValueError
            for item, result in zip(items, results, strict=True):
                if (
                    not isinstance(result, tuple)
                    or len(result) != 5
                    or result[1:] != (spec.dims, "local", spec.name, target_size)
                ):
                    raise ValueError
                name, variant = self.inputs[item.image_base64]
                expected = np.asarray(
                    self.references[spec.name][name][variant], dtype=float
                )
                actual = np.asarray(result[0], dtype=float)
                if item.normalize:
                    expected = expected / max(float(np.linalg.norm(expected)), 1e-12)
                if (
                    expected.shape != (spec.dims,)
                    or actual.shape != expected.shape
                    or not np.isfinite(expected).all()
                    or not np.isfinite(actual).all()
                    or not np.allclose(actual, expected, rtol=1e-4, atol=1e-4)
                ):
                    raise ValueError
        except InferenceCancelled:
            # No complete batch result exists; do not claim parity or native failure.
            raise
        except Exception:
            trace.verify_native_result(False)
            # Do not let numeric assertion helpers or native errors log private data.
            raise AssertionError(
                "Representative native batch failed validation"
            ) from None
        trace.verify_native_result(True)
        return results
