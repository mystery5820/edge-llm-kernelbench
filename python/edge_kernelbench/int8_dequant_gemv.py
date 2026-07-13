"""
INT8 Dequant-GEMV 的 PyTorch 参考实现。

目标计算：

    dequant_weight = weight_int8.float() * scale[:, None]
    output = x @ dequant_weight.T + bias

其中：

    x:
        浮点输入，形状 [..., in_features]

    weight_int8:
        INT8 权重，形状 [out_features, in_features]

    scale:
        每个输出通道一个反量化 scale，形状 [out_features]

    bias:
        可选浮点 bias，形状 [out_features]
"""

from __future__ import annotations

import torch
from torch import nn


def _select_compute_dtype(
    x: torch.Tensor,
) -> torch.dtype:
    """
    选择内部计算 dtype。
    """

    if x.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return torch.float32

    return x.dtype


def int8_dequant_gemv_reference(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    使用 PyTorch 实现 INT8 Dequant-GEMV。
    """

    if not x.is_floating_point():
        raise TypeError(
            "x must be a floating-point tensor, "
            f"but received dtype={x.dtype}"
        )

    if weight_int8.dtype != torch.int8:
        raise TypeError(
            "weight_int8 must be torch.int8, "
            f"but received dtype={weight_int8.dtype}"
        )

    if not scale.is_floating_point():
        raise TypeError(
            "scale must be a floating-point tensor, "
            f"but received dtype={scale.dtype}"
        )

    if bias is not None and not bias.is_floating_point():
        raise TypeError(
            "bias must be a floating-point tensor when provided, "
            f"but received dtype={bias.dtype}"
        )

    if x.ndim < 1:
        raise ValueError(
            "x must have at least one dimension"
        )

    if weight_int8.ndim != 2:
        raise ValueError(
            "weight_int8 must be a 2-dimensional tensor, "
            f"but received shape={tuple(weight_int8.shape)}"
        )

    if scale.ndim != 1:
        raise ValueError(
            "scale must be a 1-dimensional tensor, "
            f"but received shape={tuple(scale.shape)}"
        )

    if bias is not None and bias.ndim != 1:
        raise ValueError(
            "bias must be a 1-dimensional tensor, "
            f"but received shape={tuple(bias.shape)}"
        )

    out_features, in_features = weight_int8.shape

    if x.shape[-1] != in_features:
        raise ValueError(
            "the last dimension of x must match weight_int8.shape[1]: "
            f"x.shape[-1]={x.shape[-1]}, "
            f"weight_int8.shape[1]={in_features}"
        )

    if scale.numel() != out_features:
        raise ValueError(
            "scale.numel() must match weight_int8.shape[0]: "
            f"scale.numel()={scale.numel()}, "
            f"weight_int8.shape[0]={out_features}"
        )

    if bias is not None and bias.numel() != out_features:
        raise ValueError(
            "bias.numel() must match weight_int8.shape[0]: "
            f"bias.numel()={bias.numel()}, "
            f"weight_int8.shape[0]={out_features}"
        )

    tensors = [
        ("weight_int8", weight_int8),
        ("scale", scale),
    ]

    if bias is not None:
        tensors.append(
            ("bias", bias)
        )

    for name, tensor in tensors:
        if tensor.device != x.device:
            raise ValueError(
                f"{name} and x must be on the same device: "
                f"{name}.device={tensor.device}, x.device={x.device}"
            )

    compute_dtype = _select_compute_dtype(
        x
    )

    x_compute = x.to(
        dtype=compute_dtype,
    )

    weight_compute = weight_int8.to(
        dtype=compute_dtype,
    )

    scale_compute = scale.to(
        dtype=compute_dtype,
    )

    dequant_weight = (
        weight_compute
        * scale_compute[:, None]
    )

    output_compute = torch.matmul(
        x_compute,
        dequant_weight.transpose(
            0,
            1,
        ),
    )

    if bias is not None:
        output_compute = (
            output_compute
            + bias.to(
                dtype=compute_dtype,
            )
        )

    return output_compute.to(
        dtype=x.dtype,
    )


class INT8DequantGEMVReference(nn.Module):
    """
    INT8 Dequant-GEMV 的 nn.Module 封装。
    """

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if weight_int8.dtype != torch.int8:
            raise TypeError(
                "weight_int8 must be torch.int8"
            )

        self.register_buffer(
            "weight_int8",
            weight_int8
        )

        self.scale = nn.Parameter(
            scale
        )

        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(
                bias
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 INT8 Dequant-GEMV。
        """

        return int8_dequant_gemv_reference(
            x,
            self.weight_int8,
            self.scale,
            self.bias,
        )
