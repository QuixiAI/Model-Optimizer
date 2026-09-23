# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for performance measurement."""

import time
from contextlib import ContextDecorator

from . import distributed as dist
from .device import (
    DeviceLike,
    accelerator_empty_cache,
    accelerator_synchronize,
    get_accelerator_memory_info,
    get_accelerator_memory_stats,
    is_accelerator_device,
    resolve_device,
)
from .logging import print_rank_0

__all__ = [
    "AccumulatingTimer",
    "Timer",
    "clear_cuda_cache",
    "get_accelerator_memory_stats",
    "get_cuda_memory_stats",
    "get_used_gpu_mem_fraction",
    "maybe_clear_cuda_cache",
    "report_memory",
]


def clear_cuda_cache():
    """Clear the current accelerator cache.

    The CUDA-specific name is retained for backward compatibility.
    """
    accelerator_empty_cache()


_EMPTY_CACHE_CHECK_EVERY = 64
_empty_cache_calls = 0


def maybe_clear_cuda_cache(slack_bytes: int = 4 * 1024**3, device: DeviceLike = "auto") -> None:
    """Clear the accelerator cache periodically when there is enough slack to reclaim.

    ``empty_cache()`` syncs the device and hands cached blocks back to the driver, so calling it
    after every packed weight costs more than it saves. The counter restarts on each check, so the
    interval is measured from the last one rather than from process start. A caller that makes
    fewer than ``_EMPTY_CACHE_CHECK_EVERY`` calls may not reclaim at all -- use
    :func:`clear_cuda_cache` if you need a guaranteed one.
    """
    global _empty_cache_calls
    _empty_cache_calls += 1
    if _empty_cache_calls < _EMPTY_CACHE_CHECK_EVERY:
        return
    _empty_cache_calls = 0
    if not is_accelerator_device(device):
        return
    device = resolve_device(device)
    stats = get_accelerator_memory_stats(device)
    if stats.get("reserved", 0) - stats.get("allocated", 0) > slack_bytes:
        accelerator_empty_cache(device)


def get_cuda_memory_stats(device=None):
    """Get accelerator memory usage in bytes.

    The CUDA-specific name is retained for backward compatibility.
    """
    return get_accelerator_memory_stats(device)


def get_used_gpu_mem_fraction(device: DeviceLike = "auto"):
    """Get used accelerator memory as a fraction of total memory.

    Args:
        device: Device identifier. Defaults to the current accelerator.

    Returns:
        float: Fraction of GPU memory currently used (0.0 to 1.0).
               Returns 0.0 if memory reporting is unavailable.
    """
    memory_info = get_accelerator_memory_info(device)
    if memory_info is None:
        return 0.0

    free_memory, total_memory = memory_info
    return (total_memory - free_memory) / total_memory if total_memory > 0 else 0.0


def report_memory(name="", rank=0, device=None):
    """Print a simple accelerator memory report."""
    memory_stats = get_accelerator_memory_stats(device)
    string = name + " memory (MB)"
    for k, v in memory_stats.items():
        string += f" | {k}: {v / 2**20: .2e}"

    if dist.is_initialized():
        string = f"[Rank {dist.rank()}] " + string
        if dist.rank() == rank:
            print(string, flush=True)
    else:
        print(string, flush=True)


class Timer(ContextDecorator):
    """A Timer that can be used as a decorator as well."""

    def __init__(self, name="", device: DeviceLike = "auto"):
        """Initialize Timer."""
        super().__init__()
        self.name = name
        self.device = resolve_device(device)
        self._start_time = 0.0
        self.estimated_time = 0
        self.start()

    def start(self):
        """Start the timer."""
        accelerator_synchronize(self.device)
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """End the timer."""
        accelerator_synchronize(self.device)
        self.estimated_time = (time.perf_counter() - self._start_time) * 1000
        return self.estimated_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        print_rank_0(f"{self.name} took {self.estimated_time:.3e} ms")


class AccumulatingTimer(ContextDecorator):
    """A timer that accumulates time across multiple calls and works for both CUDA and non-CUDA operations."""

    # Class-level dictionary to store accumulated times by name
    _accumulated_times = {}
    _call_counts = {}
    _prefix = []

    def __init__(self, name="", device: DeviceLike = "auto"):
        """Initialize AccumulatingTimer.

        Args:
            name: Name of the timer for reporting
            device: Device whose queued work should be synchronized before timing.
        """
        super().__init__()
        self.name = name
        self.device = resolve_device(device)
        self._start_time = None

    def start(self) -> None:
        """Start the timer."""
        accelerator_synchronize(self.device)
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """End the timer and return the elapsed time in milliseconds."""
        accelerator_synchronize(self.device)
        elapsed_time = (time.perf_counter() - self._start_time) * 1000  # in milliseconds

        # Update the accumulated time and call count
        name = self.name if not AccumulatingTimer._prefix else "->".join(AccumulatingTimer._prefix)
        if name not in AccumulatingTimer._accumulated_times:
            AccumulatingTimer._accumulated_times[name] = 0.0
            AccumulatingTimer._call_counts[name] = 0
        AccumulatingTimer._accumulated_times[name] += elapsed_time
        AccumulatingTimer._call_counts[name] += 1

        return elapsed_time

    @classmethod
    def get_total_time(cls, name):
        """Get the total accumulated time for a timer in milliseconds."""
        return cls._accumulated_times.get(name, 0.0)

    @classmethod
    def get_call_count(cls, name):
        """Get the number of calls for a timer."""
        return cls._call_counts.get(name, 0)

    @classmethod
    def reset(cls):
        """Reset the accumulated times and call counts."""
        cls._accumulated_times = {}
        cls._call_counts = {}

    @classmethod
    def report(cls):
        """Report the accumulated times and call counts."""
        for name, t in cls._accumulated_times.items():
            print(f"{name}: {t:0.3f} ms (avg: {t / cls._call_counts[name]:0.3f} ms)")

    def __enter__(self):
        AccumulatingTimer._prefix.append(self.name)
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        AccumulatingTimer._prefix.pop()
