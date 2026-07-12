"""
RMSNorm Warp Shuffle CUDA Kernel 自动化测试。

本文件重点验证：

1. Warp CUDA Kernel 与 PyTorch Reference 的数值一致性；
2. Warp CUDA Kernel 与 Naive CUDA Kernel 的数值一致性；
3. 不同维度和 hidden_size 下的计算正确性；
4. 全零输入等特殊输入的正确性；
5. 输出 shape、dtype、device 和内存连续性；
6. 空行输入能否安全返回。

公共输入检查已经由 tests/test_rmsnorm_cuda.py 覆盖，
因此本文件主要关注 Warp 实现本身。
"""

from pathlib import Path
import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# 项目路径处理
# ---------------------------------------------------------------------------

# 当前文件路径：
#
#     edge-llm-kernelbench/
#     └── tests/
#         └── test_rmsnorm_warp.py
#
# parents[1] 对应项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Python 源码目录：
#
#     edge-llm-kernelbench/python
PYTHON_DIRECTORY = PROJECT_ROOT / "python"

# 将 python/ 加入模块搜索路径。
if str(PYTHON_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIRECTORY),
    )


from edge_kernelbench.rmsnorm import rmsnorm_reference
from edge_kernelbench.rmsnorm_cuda import (
    rmsnorm_cuda,
    rmsnorm_cuda_warp,
)


# ---------------------------------------------------------------------------
# CUDA 环境要求
# ---------------------------------------------------------------------------

# 本文件测试的是 CUDA Kernel。
#
# 在没有 CUDA 的设备上运行时，
# 跳过整个测试模块，而不是直接失败。
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RMSNorm Warp CUDA tests require a CUDA-capable device",
)


# ---------------------------------------------------------------------------
# 公共正确性验证函数
# ---------------------------------------------------------------------------


def assert_warp_matches_other_implementations(
    shape: tuple[int, ...],
    eps: float = 1e-6,
) -> None:
    """
    对指定形状创建随机输入，并比较三种实现：

        1. PyTorch Reference
        2. Naive CUDA Kernel
        3. Warp Shuffle CUDA Kernel

    参数
    ----------
    shape:
        输入张量形状。

        RMSNorm 始终沿最后一个维度执行。

    eps:
        数值稳定常数。
    """

    # 固定随机种子，使测试能够稳定复现。
    torch.manual_seed(2026)

    hidden_size = shape[-1]

    # 创建 CUDA FP32 输入。
    x = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float32,
    )

    # 创建一维 CUDA FP32 权重。
    weight = torch.randn(
        hidden_size,
        device="cuda",
        dtype=torch.float32,
    )

    # PyTorch Reference。
    reference_output = rmsnorm_reference(
        x,
        weight,
        eps,
    )

    # Naive Shared Memory Reduce CUDA 实现。
    naive_output = rmsnorm_cuda(
        x,
        weight,
        eps,
    )

    # Warp Shuffle Reduce CUDA 实现。
    warp_output = rmsnorm_cuda_warp(
        x,
        weight,
        eps,
    )

    # CUDA 操作默认异步执行。
    #
    # 在比较结果前进行同步，
    # 确保所有 Kernel 已经执行完成。
    torch.cuda.synchronize()

    # Warp 实现必须与 Reference 在容差范围内一致。
    torch.testing.assert_close(
        warp_output,
        reference_output,
        rtol=1e-5,
        atol=1e-6,
    )

    # Warp 实现也应与 Naive CUDA 版本一致。
    #
    # 两种 CUDA 规约顺序不同，
    # 因此允许存在非常小的浮点误差。
    torch.testing.assert_close(
        warp_output,
        naive_output,
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 固定数值测试
# ---------------------------------------------------------------------------


def test_warp_known_values() -> None:
    """
    使用容易人工理解的固定输入验证 Warp Kernel。
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

    actual = rmsnorm_cuda_warp(
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
        # 一维输入。
        (1024,),

        # hidden_size 小于线程块大小 256。
        (4, 128),

        # hidden_size 不是 2 的整数次幂。
        (2, 3, 257),

        # 常见二维输入。
        (8, 1024),

        # 单行大隐藏维度。
        (1, 4096),

        # 多行大隐藏维度。
        (16, 4096),

        # 更大 rows，用于覆盖多个 CUDA Block。
        (128, 4096),
    ],
)
def test_warp_matches_reference_and_naive(
    shape: tuple[int, ...],
) -> None:
    """
    验证多种输入形状下的三方数值一致性。
    """

    assert_warp_matches_other_implementations(
        shape
    )


# ---------------------------------------------------------------------------
# 特殊输入测试
# ---------------------------------------------------------------------------


def test_warp_zero_input() -> None:
    """
    验证全零输入。

    当输入全为零时：

        mean_square = 0

        inverse_rms = 1 / sqrt(eps)

    但最终输出仍然为零，因为：

        0 * inverse_rms * weight = 0
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

    output = rmsnorm_cuda_warp(
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


def test_warp_constant_input() -> None:
    """
    验证所有元素相同的输入。

    这种输入便于检查平方和规约是否完整，
    可以防止漏加某些线程负责的元素。
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

    actual = rmsnorm_cuda_warp(
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


def test_warp_output_metadata() -> None:
    """
    Warp Kernel 输出应继承输入的：

    - shape
    - dtype
    - device

    并保持连续内存布局。
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

    output = rmsnorm_cuda_warp(
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


def test_warp_empty_rows() -> None:
    """
    验证形状为 [0, hidden_size] 的输入。

    此时：

        rows = 0

    Launcher 不应启动 CUDA Kernel，
    而应直接返回空输出。
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

    output = rmsnorm_cuda_warp(
        x,
        weight,
    )

    assert output.shape == x.shape
    assert output.numel() == 0
    assert output.dtype == torch.float32
    assert output.device.type == "cuda"