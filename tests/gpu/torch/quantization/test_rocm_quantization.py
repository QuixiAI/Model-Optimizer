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

"""Exercise the portable quantization paths on ROCm devices."""

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.qtensor import NVFP4QTensor

pytestmark = pytest.mark.skipif(torch.version.hip is None, reason="Requires ROCm PyTorch")


@pytest.mark.parametrize(
    "config_name", ["FP8_DEFAULT_CFG", "MXFP8_DEFAULT_CFG", "NVFP4_DEFAULT_CFG"]
)
def test_fake_quantization_on_rocm(config_name):
    model = torch.nn.Sequential(torch.nn.Linear(128, 128)).to("cuda")
    inputs = torch.randn(2, 128, device="cuda")

    mtq.quantize(model, getattr(mtq, config_name), lambda module: module(inputs))

    output = model(inputs)
    assert output.shape == (2, 128)
    assert torch.isfinite(output).all()

    mtq.compress(model)
    assert torch.isfinite(model(inputs)).all()


def test_nvfp4_fast_dequantize_uses_portable_path_on_rocm():
    weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    packed, scale, double_scale = NVFP4QTensor.quantize(weight, block_size=16)
    kwargs = {"scale": scale, "double_scale": double_scale, "block_sizes": {-1: 16}}

    expected = packed.dequantize(fast=False, **kwargs)
    result = packed.dequantize(fast=True, **kwargs)

    torch.testing.assert_close(result, expected)
