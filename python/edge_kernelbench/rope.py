"""
RoPE 的 PyTorch 参考实现。

RoPE 全称 Rotary Position Embedding，即旋转位置编码。
它常用于 Transformer Attention 中的 query 和 key。

对最后一个维度按 even / odd 两两成对旋转：

    y_even = x_even * cos - x_odd * sin
    y_odd  = x_even * sin + x_odd * cos

本文件的作用：

1. 作为后续 CUDA RoPE Kernel 的正确性基准；
2. 提供清晰的输入检查；
3. 支持 CPU / CUDA；
4. 支持常见形状：

       [seq_len, num_heads, head_dim]
       [batch_size, seq_len, num_heads, head_dim]

   也支持更简单的：

       [seq_len, head_dim]

5. FP16 / BF16 输入内部使用 FP32 计算，输出恢复为输入 dtype。
"""

from __future__ import annotations

import torch
from torch import nn


def _check_floating_tensor(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    检查张量是否为浮点类型。
    """

    if not tensor.is_floating_point():
        raise TypeError(
            f"{name} must be a floating-point tensor, "
            f"but received dtype={tensor.dtype}"
        )


def _select_compute_dtype(
    x: torch.Tensor,
) -> torch.dtype:
    """
    选择 RoPE 内部计算 dtype。

    FP16 / BF16 在乘法和加减过程中更容易积累误差，
    因此内部提升到 FP32。
    """

    if x.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return torch.float32

    return x.dtype


def _reshape_trig_for_rope(
    x: torch.Tensor,
    trig: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """
    将 cos / sin reshape 成可以与 x 的 even/odd 部分广播的形状。

    x 的最后一维是 head_dim，RoPE 每两个元素组成一对，
    因此 cos / sin 的最后一维必须是 head_dim / 2。

    支持三类常见输入：

    1. [head_dim / 2]
       对所有 token / head 使用同一组旋转参数；

    2. [seq_len, head_dim / 2]
       按 token 位置提供旋转参数；

    3. 与 x[..., 0::2] 可直接广播的形状。
    """

    half_dim = x.shape[-1] // 2
    x_even_shape = x[..., 0::2].shape

    if trig.ndim == 0:
        raise ValueError(
            f"{name} must have at least one dimension"
        )

    if trig.shape[-1] != half_dim:
        raise ValueError(
            f"the last dimension of {name} must be head_dim / 2: "
            f"{name}.shape[-1]={trig.shape[-1]}, "
            f"head_dim / 2={half_dim}"
        )

    if trig.ndim == 1:
        return trig.reshape(
            *([1] * (x.ndim - 1)),
            half_dim,
        )

    if trig.shape == x_even_shape:
        return trig

    if trig.ndim == 2:
        seq_len = trig.shape[0]

        if x.ndim == 2:
            if x.shape[0] != seq_len:
                raise ValueError(
                    f"{name}.shape[0] must match x.shape[0] for 2D x: "
                    f"{name}.shape[0]={seq_len}, "
                    f"x.shape[0]={x.shape[0]}"
                )

            return trig

        if x.ndim < 3:
            raise ValueError(
                f"{name} with shape [seq_len, head_dim / 2] "
                "requires x to have at least 3 dimensions"
            )

        if x.shape[-3] != seq_len:
            raise ValueError(
                f"{name}.shape[0] must match the sequence dimension "
                f"x.shape[-3]: {name}.shape[0]={seq_len}, "
                f"x.shape[-3]={x.shape[-3]}"
            )

        return trig.reshape(
            *([1] * (x.ndim - 3)),
            seq_len,
            1,
            half_dim,
        )

    try:
        torch.broadcast_shapes(
            x_even_shape,
            trig.shape,
        )
    except RuntimeError as error:
        raise ValueError(
            f"{name} with shape {tuple(trig.shape)} cannot broadcast "
            f"to x even/odd shape {tuple(x_even_shape)}"
        ) from error

    return trig


def apply_rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    对单个张量应用 RoPE。

    Args:
        x:
            输入张量，最后一维为 head_dim。
            head_dim 必须是偶数。

        cos:
            旋转角的 cos 值。
            常见形状为 [seq_len, head_dim / 2]。

        sin:
            旋转角的 sin 值。
            形状必须与 cos 兼容。

    Returns:
        应用 RoPE 后的张量，shape、dtype、device 与 x 一致。
    """

    _check_floating_tensor(
        x,
        "x",
    )

    _check_floating_tensor(
        cos,
        "cos",
    )

    _check_floating_tensor(
        sin,
        "sin",
    )

    if x.ndim < 1:
        raise ValueError(
            "x must have at least one dimension"
        )

    head_dim = x.shape[-1]

    if head_dim <= 0:
        raise ValueError(
            "the last dimension of x must be greater than zero"
        )

    if head_dim % 2 != 0:
        raise ValueError(
            f"the last dimension of x must be even, but received {head_dim}"
        )

    if cos.device != x.device:
        raise ValueError(
            "cos and x must be on the same device: "
            f"cos.device={cos.device}, x.device={x.device}"
        )

    if sin.device != x.device:
        raise ValueError(
            "sin and x must be on the same device: "
            f"sin.device={sin.device}, x.device={x.device}"
        )

    compute_dtype = _select_compute_dtype(
        x
    )

    x_compute = x.to(
        dtype=compute_dtype,
    )

    cos_compute = cos.to(
        dtype=compute_dtype,
    )

    sin_compute = sin.to(
        dtype=compute_dtype,
    )

    cos_broadcast = _reshape_trig_for_rope(
        x_compute,
        cos_compute,
        "cos",
    )

    sin_broadcast = _reshape_trig_for_rope(
        x_compute,
        sin_compute,
        "sin",
    )

    if cos_broadcast.shape != sin_broadcast.shape:
        try:
            torch.broadcast_shapes(
                cos_broadcast.shape,
                sin_broadcast.shape,
            )
        except RuntimeError as error:
            raise ValueError(
                "cos and sin shapes are not broadcast-compatible: "
                f"cos.shape={tuple(cos.shape)}, "
                f"sin.shape={tuple(sin.shape)}"
            ) from error

    x_even = x_compute[..., 0::2]
    x_odd = x_compute[..., 1::2]

    output_even = (
        x_even
        * cos_broadcast
        - x_odd
        * sin_broadcast
    )

    output_odd = (
        x_even
        * sin_broadcast
        + x_odd
        * cos_broadcast
    )

    output = torch.empty_like(
        x_compute
    )

    output[..., 0::2] = output_even
    output[..., 1::2] = output_odd

    return output.to(
        dtype=x.dtype,
    )


def rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    同时对 query 和 key 应用 RoPE。
    """

    if q.shape != k.shape:
        raise ValueError(
            "q and k must have the same shape: "
            f"q.shape={tuple(q.shape)}, "
            f"k.shape={tuple(k.shape)}"
        )

    if q.device != k.device:
        raise ValueError(
            "q and k must be on the same device: "
            f"q.device={q.device}, k.device={k.device}"
        )

    if q.dtype != k.dtype:
        raise TypeError(
            "q and k must have the same dtype: "
            f"q.dtype={q.dtype}, k.dtype={k.dtype}"
        )

    q_output = apply_rope_reference(
        q,
        cos,
        sin,
    )

    k_output = apply_rope_reference(
        k,
        cos,
        sin,
    )

    return (
        q_output,
        k_output,
    )


class RoPEReference(nn.Module):
    """
    RoPE 的 nn.Module 封装。
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 q 和 k 应用 RoPE。
        """

        return rope_reference(
            q,
            k,
            cos,
            sin,
        )
