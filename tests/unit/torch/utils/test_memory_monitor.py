# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch

from modelopt.torch.utils import memory_monitor


def test_launch_memory_monitor_returns_none_on_cpu():
    assert memory_monitor.launch_memory_monitor(device="cpu") is None


def _fake_xpu(monkeypatch):
    monkeypatch.setattr(memory_monitor, "resolve_device", lambda device: torch.device("xpu"))
    monkeypatch.setattr(memory_monitor, "get_accelerator_device_count", lambda device: 2)
    monkeypatch.setattr(
        memory_monitor,
        "with_device_index",
        lambda device, index: torch.device("xpu", index),
    )
    monkeypatch.setattr(
        memory_monitor,
        "get_accelerator_memory_info",
        lambda device: (25, 100) if device.index == 0 else (50, 100),
    )


def test_monitor_collects_all_devices_without_vendor_library(monkeypatch):
    _fake_xpu(monkeypatch)

    monitor = memory_monitor.AcceleratorMemoryMonitor(device="xpu")
    monkeypatch.setattr(
        memory_monitor.time,
        "sleep",
        lambda interval: setattr(monitor, "is_running", False),
    )
    monitor.is_running = True
    monitor._monitor_loop()

    assert monitor.peak_memory == {0: 75 / 2**30, 1: 50 / 2**30}


def test_stopping_a_stopped_monitor_does_nothing(monkeypatch, capsys):
    """launch_memory_monitor registers stop() to run at exit, so a monitor its caller stopped is
    stopped again then: the second stop does not report again."""
    _fake_xpu(monkeypatch)
    monitor = memory_monitor.launch_memory_monitor(monitor_interval=0.01, device="xpu")

    monitor.stop()
    assert "########" in capsys.readouterr().out
    monitor.stop()

    assert capsys.readouterr().out == ""


def test_legacy_gpu_monitor_name_is_preserved():
    assert memory_monitor.GPUMemoryMonitor is memory_monitor.AcceleratorMemoryMonitor
