# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, PLR0917

"""Packed NVFP4 weight GEMM for ROCm using BF16/FP16 matrix instructions."""

import torch
from torch.autograd import Function

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.backends.gemm_registry import gemm_registry
from modelopt.torch.quantization.backends.utils import quantizer_matches_default_cfg
from modelopt.torch.quantization.qtensor import NVFP4QTensor, QTensorWrapper

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

__all__ = ["RocmNvfp4Linear", "rocm_nvfp4_gemm"]


if triton is not None:

    @triton.jit
    def _packed_nvfp4_gemm(
        X,
        W,
        S,
        D,
        Y,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        mi = tl.program_id(0) * BM + tl.arange(0, BM)
        ni = tl.program_id(1) * BN + tl.arange(0, BN)
        ki = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        double_scale = tl.load(D).to(tl.float32)

        for start in range(tl.cdiv(K, BK)):
            k = start * BK + ki
            x = tl.load(X + mi[:, None] * K + k[None, :], (mi[:, None] < M) & (k[None, :] < K), 0)
            packed = tl.load(
                W + ni[:, None] * (K // 2) + k[None, :] // 2,
                (ni[:, None] < N) & (k[None, :] < K),
                0,
            ).to(tl.int32)
            nibble = tl.where(k[None, :] % 2 == 0, packed & 15, packed >> 4)
            code = nibble & 7
            value = tl.where(code == 0, 0.0, 6.0)
            value = tl.where(code == 1, 0.5, value)
            value = tl.where(code == 2, 1.0, value)
            value = tl.where(code == 3, 1.5, value)
            value = tl.where(code == 4, 2.0, value)
            value = tl.where(code == 5, 3.0, value)
            value = tl.where(code == 6, 4.0, value)
            value = tl.where(nibble & 8 != 0, -value, value)
            scale_bits = tl.load(
                S.to(tl.pointer_type(tl.uint8)) + ni[:, None] * (K // 16) + k[None, :] // 16,
                (ni[:, None] < N) & (k[None, :] < K),
                0,
            )
            scale = scale_bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            weight = (value * scale * double_scale).to(x.dtype)
            acc += tl.dot(x, tl.trans(weight))

        tl.store(Y + mi[:, None] * N + ni[None, :], acc, (mi[:, None] < M) & (ni[None, :] < N))


def rocm_nvfp4_gemm(quant_module, input_tensor, bias=None):
    """Multiply fake-quantized activations by packed NVFP4 weights on ROCm."""
    weight = quant_module.weight.get_qtensor()
    x = quant_module.input_quantizer(input_tensor).contiguous()
    shape = x.shape
    m, k = x.numel() // shape[-1], shape[-1]
    if m > 256:
        dense_weight = weight.dequantize(
            scale=quant_module.weight_quantizer._scale,
            double_scale=quant_module.weight_quantizer._double_scale,
            block_sizes={-1: 16},
        )
        return torch.nn.functional.linear(x, dense_weight, bias)

    n = weight.metadata["shape"][0]
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    block_m = 64 if m >= 64 else 16
    _packed_nvfp4_gemm[(triton.cdiv(m, block_m), triton.cdiv(n, 64))](
        x.reshape(m, k),
        weight._quantized_data,
        quant_module.weight_quantizer._scale,
        quant_module.weight_quantizer._double_scale,
        output,
        m,
        n,
        k,
        BM=block_m,
        BN=64,
        BK=64,
        num_warps=4,
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*shape[:-1], n)


class RocmNvfp4Linear(Function):
    """Autograd wrapper for the ROCm packed NVFP4 GEMM."""

    @staticmethod
    def forward(
        ctx, quant_module, input_tensor, weight, bias=None, allreduce_dgrad=False, tp_group=None
    ):
        """Run the packed GEMM and retain tensors for the portable backward path."""
        ctx.save_for_backward(
            input_tensor if weight.requires_grad else None,
            weight if input_tensor.requires_grad else None,
            getattr(quant_module.weight_quantizer, "_scale", None),
            getattr(quant_module.weight_quantizer, "_double_scale", None),
        )
        ctx.compute_bias_grad = bias is not None and bias.requires_grad
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        return rocm_nvfp4_gemm(quant_module, input_tensor, bias)

    @staticmethod
    def backward(ctx, grad_outputs):
        """Use dequantized weights for gradients, as in the XPU backend."""
        input_tensor, weight, scale, double_scale = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if weight is not None:
            if isinstance(weight, QTensorWrapper):
                weight = weight.get_qtensor().dequantize(
                    scale=scale, double_scale=double_scale, block_sizes={-1: 16}
                )
            grad_input = grad_outputs @ weight
        if input_tensor is not None:
            grad_weight = grad_outputs.reshape(-1, grad_outputs.shape[-1]).T @ input_tensor.reshape(
                -1, input_tensor.shape[-1]
            )
        if ctx.compute_bias_grad:
            grad_bias = grad_outputs.sum(dim=list(range(grad_outputs.dim() - 1)))
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(grad_input, group=ctx.tp_group)
        return None, grad_input, grad_weight, grad_bias, None, None

    @classmethod
    def apply(cls, *args, **kwargs):
        """Pass keyword arguments through the autograd function."""
        return super().apply(*args, *tuple(kwargs.values()))


def _rocm_nvfp4_availability_check(module, input, args, kwargs):
    if triton is None or torch.version.hip is None or input.device.type != "cuda":
        return False
    if input.dtype not in (torch.bfloat16, torch.float16):
        return False
    from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear

    if not isinstance(module, RealQuantLinear):
        return False
    if not quantizer_matches_default_cfg(module, mtq.NVFP4_DEFAULT_CFG):
        return False
    if not isinstance(module.weight, QTensorWrapper) or not isinstance(
        module.weight.get_qtensor(), NVFP4QTensor
    ):
        return False
    scale = getattr(module.weight_quantizer, "_scale", None)
    double_scale = getattr(module.weight_quantizer, "_double_scale", None)
    return (
        input.shape[-1] % 16 == 0
        and scale is not None
        and scale.dtype == torch.float8_e4m3fn
        and scale.ndim == 2
        and scale.shape == (module.weight.shape[0], input.shape[-1] // 16)
        and double_scale is not None
        and double_scale.numel() == 1
    )


gemm_registry.register(
    gemm_func=RocmNvfp4Linear.apply,
    availability_check=_rocm_nvfp4_availability_check,
)
