"""
RMSNorm float4 向量化 CUDA Kernel 自动化测试。

本文件重点验证：

1. float4 CUDA Kernel 与 PyTorch Reference 的数值一致性；
2. float4 CUDA Kernel 与 Naive/Warp CUDA Kernel 的数值一致性；
3. hidden_size 不能被 4 整除时尾部标量处理正确；
4. 输入首地址不满足 16 字节对齐时可以安全回退；
5. 输出 shape、dtype、device 和连续性正确；
6. 空行输入能安全返回。
"""

from pathlib import Path
import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# 项目路径处理
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIRECTORY = PROJECT_ROOT / "python"

if str(PYTHON_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIRECTORY),
    )


from edge_kernelbench.rmsnorm import rmsnorm_reference
from edge_kernelbench.rmsnorm_cuda import (
    rmsnorm_cuda,
    rmsnorm_cuda_float4,
    rmsnorm_cuda_warp,
)


# ---------------------------------------------------------------------------
# CUDA 环境要求
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RMSNorm float4 CUDA tests require a CUDA-capable device",
)


# ---------------------------------------------------------------------------
# 公共正确性验证函数
# ---------------------------------------------------------------------------


def assert_float4_matches_other_implementations(
    shape: tuple[int, ...],
    eps: float = 1e-6,
) -> None:
    """
    对指定形状创建随机输入，并比较四种实现：

        1. PyTorch Reference
        2. Naive CUDA Kernel
        3. Warp Shuffle CUDA Kernel
        4. float4 CUDA Kernel
    """

    torch.manual_seed(2026)

    hidden_size = shape[-1]

    x = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.randn(
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    )

    reference_output = rmsnorm_reference(
        x,
        weight,
        eps,
    )

    naive_output = rmsnorm_cuda(
        x,
        weight,
        eps,
    )

    warp_output = rmsnorm_cuda_warp(
        x,
        weight,
        eps,
    )

    float4_output = rmsnorm_cuda_float4(
        x,
        weight,
        eps,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        float4_output,
        reference_output,
        rtol=1e-5,
        atol=1e-6,
    )

    torch.testing.assert_close(
        float4_output,
        naive_output,
        rtol=1e-5,
        atol=1e-6,
    )

    torch.testing.assert_close(
        float4_output,
        warp_output,
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 固定数值测试
# ---------------------------------------------------------------------------


def test_float4_known_values() -> None:
    """
    使用容易人工理解的固定输入验证 float4 Kernel。
    """

    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        4,
        device="cuda",
        dtype=torch.float32,
    )

    expected = rmsnorm_reference(
        x,
        weight,
        eps=1e-6,
    )

    actual = rmsnorm_cuda_float4(
        x,
        weight,
        eps=1e-6,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 多种形状测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1024,),
        (4, 128),
        (2, 3, 257),
        (1, 4096),
        (16, 4096),
        (128, 4096),
    ],
)
def test_float4_matches_reference_naive_and_warp(
    shape: tuple[int, ...],
) -> None:
    """
    验证常见形状下的四方数值一致性。
    """

    assert_float4_matches_other_implementations(
        shape
    )


@pytest.mark.parametrize(
    "hidden_size",
    [
        1025,
        1026,
        1027,
    ],
)
def test_float4_scalar_tail_hidden_sizes(
    hidden_size: int,
) -> None:
    """
    验证 hidden_size % 4 != 0 时尾部标量处理正确。
    """

    assert_float4_matches_other_implementations(
        (3, hidden_size)
    )


# ---------------------------------------------------------------------------
# 特殊输入测试
# ---------------------------------------------------------------------------


def test_float4_zero_input() -> None:
    """
    验证全零输入。
    """

    x = torch.zeros(
        8,
        4096,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.randn(
        4096,
        device="cuda",
        dtype=torch.float32,
    )

    output = rmsnorm_cuda_float4(
        x,
        weight,
        eps=1e-6,
    )

    torch.cuda.synchronize()

    expected = torch.zeros_like(x)

    torch.testing.assert_close(
        output,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_float4_constant_input() -> None:
    """
    验证所有元素相同的输入。
    """

    x = torch.full(
        (4, 1024),
        fill_value=2.0,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        1024,
        device="cuda",
        dtype=torch.float32,
    )

    expected = rmsnorm_reference(
        x,
        weight,
        eps=1e-6,
    )

    actual = rmsnorm_cuda_float4(
        x,
        weight,
        eps=1e-6,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 输出属性测试
# ---------------------------------------------------------------------------


def test_float4_output_metadata() -> None:
    """
    float4 Kernel 输出应继承输入的 shape、dtype、device 和连续布局。
    """

    x = torch.randn(
        2,
        3,
        512,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        512,
        device="cuda",
        dtype=torch.float32,
    )

    output = rmsnorm_cuda_float4(
        x,
        weight,
    )

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device
    assert output.is_contiguous()


# ---------------------------------------------------------------------------
# 空输入测试
# ---------------------------------------------------------------------------


def test_float4_empty_rows() -> None:
    """
    验证形状为 [0, hidden_size] 的输入。
    """

    x = torch.empty(
        0,
        128,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        128,
        device="cuda",
        dtype=torch.float32,
    )

    output = rmsnorm_cuda_float4(
        x,
        weight,
    )

    assert output.shape == x.shape
    assert output.numel() == 0
    assert output.dtype == torch.float32
    assert output.device.type == "cuda"


# ---------------------------------------------------------------------------
# 非对齐回退测试
# ---------------------------------------------------------------------------


def test_float4_unaligned_contiguous_input_falls_back() -> None:
    """
    验证连续但首地址不满足 16 字节对齐的输入可以安全计算。

    base[1:] 会让 x 的 storage_offset 等于 1，
    对 FP32 来说首地址偏移 4 字节，不能使用 float4 访问。
    """

    torch.manual_seed(2026)

    rows = 5
    hidden_size = 1024

    base = torch.randn(
        rows * hidden_size + 1,
        device="cuda",
        dtype=torch.float32,
    )

    x = base[1:].view(
        rows,
        hidden_size,
    )

    assert x.is_contiguous()
    assert x.storage_offset() == 1

    weight = torch.randn(
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    )

    reference_output = rmsnorm_reference(
        x,
        weight,
        eps=1e-6,
    )

    float4_output = rmsnorm_cuda_float4(
        x,
        weight,
        eps=1e-6,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        float4_output,
        reference_output,
        rtol=1e-5,
        atol=1e-6,
    )
