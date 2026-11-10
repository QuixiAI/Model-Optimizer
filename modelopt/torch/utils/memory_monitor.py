# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Backend-neutral accelerator memory monitoring utilities."""

import atexit
import threading
import time

from .device import (
    DeviceLike,
    get_accelerator_device_count,
    get_accelerator_memory_info,
    resolve_device,
    with_device_index,
)


class AcceleratorMemoryMonitor:
    """Track peak memory use across devices on the selected PyTorch accelerator."""

    def __init__(self, monitor_interval: float = 10.0, device: DeviceLike = "auto"):
        """Initialize an accelerator memory monitor."""
        self.monitor_interval = monitor_interval
        self.device = resolve_device(device)
        self.device_count = get_accelerator_device_count(self.device)
        if self.device_count == 0:
            raise RuntimeError(f"No accelerator devices are available for {self.device.type!r}.")
        self.peak_memory: dict[int, float] = {}
        self.is_running = False
        self.monitor_thread: threading.Thread | None = None
        self._stopped = False

    def _monitor_loop(self):
        while self.is_running:
            for index in range(self.device_count):
                memory_info = get_accelerator_memory_info(with_device_index(self.device, index))
                if memory_info is None:
                    continue
                free_memory, total_memory = memory_info
                used_memory_gb = (total_memory - free_memory) / 2**30
                self.peak_memory[index] = max(self.peak_memory.get(index, 0), used_memory_gb)
            time.sleep(self.monitor_interval)

    def start(self):
        """Start memory monitoring in a daemon thread."""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring and print peak memory use for each device.

        Stopping a stopped monitor does nothing, as when a caller stops it and the exit hook
        :func:`launch_memory_monitor` registered runs too.
        """
        if self._stopped:
            return
        self._stopped = True
        self.is_running = False
        # The thread can be inside a sampling pass, still adding to peak_memory: let it finish first.
        if self.monitor_thread:
            self.monitor_thread.join()
        print("########")
        for device_index, peak_memory in self.peak_memory.items():
            print(
                f"{self.device.type}:{device_index}: Peak memory usage = "
                f"{peak_memory:.2f} GB for all processes on the device"
            )
        print("########")


# Backward-compatible name for callers that imported the old NVML-specific class.
GPUMemoryMonitor = AcceleratorMemoryMonitor


def launch_memory_monitor(
    monitor_interval: float = 1.0, device: DeviceLike = "auto"
) -> AcceleratorMemoryMonitor | None:
    """Create and start an accelerator memory monitor when reporting is supported."""
    try:
        resolved = resolve_device(device)
        if get_accelerator_device_count(resolved) == 0:
            return None
        if get_accelerator_memory_info(with_device_index(resolved, 0)) is None:
            return None
        monitor = AcceleratorMemoryMonitor(monitor_interval, resolved)
    except (RuntimeError, ValueError) as error:
        print(f"Failed to get accelerator memory info: {error}. Stopping memory monitor.")
        return None
    monitor.start()
    atexit.register(monitor.stop)
    return monitor
