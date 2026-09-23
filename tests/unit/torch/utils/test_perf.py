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

import pytest
import torch

from modelopt.torch.utils import perf


def test_timer_uses_backend_neutral_clock_and_sync(monkeypatch):
    timestamps = iter([2.0, 2.025])
    synchronized = []
    monkeypatch.setattr(perf.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(perf, "accelerator_synchronize", lambda device: synchronized.append(device))

    timer = perf.Timer(device="cpu")

    assert timer.stop() == pytest.approx(25.0)
    assert synchronized == [torch.device("cpu"), torch.device("cpu")]


def test_accumulating_timer_runs_on_cpu(monkeypatch):
    timestamps = iter([4.0, 4.01])
    monkeypatch.setattr(perf.time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(perf, "accelerator_synchronize", lambda device: None)
    perf.AccumulatingTimer.reset()

    with perf.AccumulatingTimer("cpu-work", device="cpu"):
        pass

    assert perf.AccumulatingTimer.get_total_time("cpu-work") == pytest.approx(10.0)
    assert perf.AccumulatingTimer.get_call_count("cpu-work") == 1


@pytest.mark.parametrize(
    ("memory_info", "expected"),
    [((25, 100), 0.75), ((0, 0), 0.0), (None, 0.0)],
)
def test_used_memory_fraction_handles_backend_reporting(monkeypatch, memory_info, expected):
    monkeypatch.setattr(perf, "get_accelerator_memory_info", lambda device: memory_info)

    assert perf.get_used_gpu_mem_fraction("cpu") == expected


def test_cuda_compatibility_aliases_use_generic_helpers(monkeypatch):
    calls = []
    stats = {"allocated": 1, "max_allocated": 2, "reserved": 3, "max_reserved": 4}
    monkeypatch.setattr(perf, "accelerator_empty_cache", lambda: calls.append("clear"))
    monkeypatch.setattr(perf, "get_accelerator_memory_stats", lambda device: stats)

    perf.clear_cuda_cache()

    assert perf.get_cuda_memory_stats("xpu:0") == stats
    assert calls == ["clear"]


@pytest.fixture
def cuda_stub(monkeypatch):
    """Stand in for the CUDA allocator: records empty_cache calls, lets tests set the slack."""

    class Stub:
        def __init__(self):
            self.reserved = 0
            self.allocated = 0
            self.empty_cache_calls = 0

    stub = Stub()
    monkeypatch.setattr(perf, "is_accelerator_device", lambda device: True)
    monkeypatch.setattr(perf, "resolve_device", lambda device: torch.device("cuda"))
    monkeypatch.setattr(
        perf,
        "get_accelerator_memory_stats",
        lambda device: {"reserved": stub.reserved, "allocated": stub.allocated},
    )

    def _empty_cache():
        stub.empty_cache_calls += 1

    monkeypatch.setattr(perf, "accelerator_empty_cache", lambda device: _empty_cache())
    # The counter is process-global; start each test from a known phase.
    monkeypatch.setattr(perf, "_empty_cache_calls", 0)
    return stub


def test_checks_once_per_interval(cuda_stub):
    """One check per ``_EMPTY_CACHE_CHECK_EVERY`` calls, and none in between."""
    cuda_stub.reserved = 8 * 1024**3  # 8 GiB of slack, well over the default threshold
    every = perf._EMPTY_CACHE_CHECK_EVERY

    for _ in range(every - 1):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 0, "cleared before reaching the interval"

    perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 1

    for _ in range(every):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 2, "should clear once per interval, not more"


def test_counter_restarts_on_each_check(cuda_stub):
    """The interval is measured from the last check, not from process start.

    Without the reset, a caller inheriting a mid-interval counter would fire early and then
    drift; with it, every caller gets a full interval between checks.
    """
    cuda_stub.reserved = 8 * 1024**3
    every = perf._EMPTY_CACHE_CHECK_EVERY

    # Land mid-interval, as a second export in the same process would.
    for _ in range(every + 5):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 1
    assert perf._empty_cache_calls == 5, "counter should restart at the check, not keep climbing"

    for _ in range(every - 5):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 2, "a full interval after the previous check"


def test_does_not_clear_below_the_slack_threshold(cuda_stub):
    """A sampled call with too little reclaimable slack leaves the cache alone."""
    cuda_stub.reserved = 4 * 1024**3
    cuda_stub.allocated = 4 * 1024**3  # no slack at all

    for _ in range(perf._EMPTY_CACHE_CHECK_EVERY * 3):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 0


def test_slack_threshold_is_configurable(cuda_stub):
    """A caller that is memory-tight can lower the bar for reclaiming."""
    cuda_stub.reserved = 1024**3  # 1 GiB of slack: under the 4 GiB default, over a 1 MiB bar

    for _ in range(perf._EMPTY_CACHE_CHECK_EVERY):
        perf.maybe_clear_cuda_cache()
    assert cuda_stub.empty_cache_calls == 0, "1 GiB should not trip the default 4 GiB threshold"

    for _ in range(perf._EMPTY_CACHE_CHECK_EVERY):
        perf.maybe_clear_cuda_cache(slack_bytes=1024**2)
    assert cuda_stub.empty_cache_calls == 1


def test_no_accelerator_is_a_noop(monkeypatch):
    """Without an accelerator the sampled call must not touch the allocator."""
    monkeypatch.setattr(perf, "is_accelerator_device", lambda device: False)
    monkeypatch.setattr(perf, "_empty_cache_calls", 0)

    def _boom(*a, **k):
        raise AssertionError("queried the allocator with no CUDA available")

    monkeypatch.setattr(perf, "get_accelerator_memory_stats", _boom)
    monkeypatch.setattr(perf, "accelerator_empty_cache", _boom)

    for _ in range(perf._EMPTY_CACHE_CHECK_EVERY * 2):
        perf.maybe_clear_cuda_cache()
