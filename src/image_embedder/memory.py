# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import gc
from typing import Optional

from .logging_config import get_logger

logger = get_logger(__name__)


def cleanup_gpu_memory(device: Optional[str] = None) -> dict:
    result = {"gc_collected": 0, "gpu_freed_mb": 0.0}
    
    result["gc_collected"] = gc.collect()
    logger.debug(f"Garbage collection: {result['gc_collected']} objects collected")
    
    try:
        import torch
        if torch.cuda.is_available():
            if device is None or str(device).startswith("cuda"):
                before = torch.cuda.memory_allocated() / (1024 * 1024)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated() / (1024 * 1024)
                result["gpu_freed_mb"] = max(0, before - after)
                logger.debug(f"GPU cache cleared: freed {result['gpu_freed_mb']:.2f} MB")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to cleanup GPU memory: {e}")
    
    return result


def get_memory_usage() -> dict:
    result = {
        "process_rss_mb": None,
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
    }
    
    try:
        import psutil
        process = psutil.Process()
        result["process_rss_mb"] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to get process memory: {e}")
    
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to get GPU memory: {e}")
    
    return result


def check_memory_health(
    max_process_mb: Optional[float] = None,
    max_gpu_mb: Optional[float] = None,
) -> tuple[bool, list[str]]:
    issues = []
    usage = get_memory_usage()
    
    if max_process_mb and usage["process_rss_mb"]:
        if usage["process_rss_mb"] > max_process_mb:
            issues.append(
                f"Process memory ({usage['process_rss_mb']:.1f} MB) exceeds limit ({max_process_mb} MB)"
            )
    
    if max_gpu_mb and usage["gpu_allocated_mb"]:
        if usage["gpu_allocated_mb"] > max_gpu_mb:
            issues.append(
                f"GPU memory ({usage['gpu_allocated_mb']:.1f} MB) exceeds limit ({max_gpu_mb} MB)"
            )
    
    return len(issues) == 0, issues


def force_cleanup() -> dict:
    logger.info("Performing forced memory cleanup")
    
    result = cleanup_gpu_memory()
    
    gc.collect(2)
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed IPC collection: {e}")
    
    logger.info(f"Forced cleanup complete: {result}")
    return result
