"""
RMSNorm Naive CUDA Kernel 自动化测试。

本文件主要验证：

1. CUDA Kernel 的数值结果是否与 PyTorch Reference 一致；
2. 一维、二维、三维输入是否都能正确处理；
3. 不同 hidden_size 是否都能正确规约；
4. 输出的形状、数据类型和设备是否正确；
5. 空输入是否能够安全处理；
6. 非法输入是否能够被 C++ 接口正确拒绝。
"""

from pathlib import Path
import sys

import pytest
import torch


# 将项目的 python/ 目录加入模块搜索路径。
#
# tests/test_rmsnorm_cuda.py
# 的上一级目录是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIRECTORY = PROJECT_ROOT / "python"

if str(PYTHON_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIRECTORY),
    )


from edge_kernelbench.rmsnorm import rmsnorm_reference
from edge_kernelbench.rmsnorm_cuda import rmsnorm_cuda


# 当前测试文件依赖 CUDA。
#
# 在没有 CUDA 的机器上运行时，
# pytest 会跳过整个测试模块，而不是直接报错。
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RMSNorm CUDA tests require a CUDA-capable device",
)


def assert_cuda_matches_reference(
    shape: tuple[int, ...],
    eps: float = 1e-6,
) -> None:
    """
    创建指定形状的随机输入，并比较：

        PyTorch Reference

    与：

        自定义 CUDA Kernel

    的计算结果。
    """

    # 固定随机种子，使测试结果能够重复。
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

    expected = rmsnorm_reference(
        x,
        weight,
        eps,
    )

    actual = rmsnorm_cuda(
        x,
        weight,
        eps,
    )

    # CUDA 默认采用异步执行。
    #
    # 同步后再进行断言，
    # 可以确保 Kernel 已经真正执行完成。
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_known_values() -> None:
    """
    使用容易人工检查的固定输入验证 CUDA Kernel。
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

    actual = rmsnorm_cuda(
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


@pytest.mark.parametrize(
    "shape",
    [
        # 一维输入，一整条向量就是一行。
        (1024,),

        # hidden_size 小于线程块大小 256。
        (4, 128),

        # hidden_size 不是 2 的整数次幂，
        # 并且大于线程块大小。
        (2, 3, 257),

        # 常见二维输入。
        (8, 1024),

        # 接近大模型隐藏维度的输入。
        (2, 4096),
    ],
)
def test_matches_reference_for_multiple_shapes(
    shape: tuple[int, ...],
) -> None:
    """
    验证不同维度和 hidden_size 下的计算正确性。
    """

    assert_cuda_matches_reference(shape)


def test_output_metadata() -> None:
    """
    输出应当继承输入的：

    - shape
    - dtype
    - device
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

    output = rmsnorm_cuda(
        x,
        weight,
    )

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device
    assert output.is_contiguous()


def test_empty_rows() -> None:
    """
    验证形状为 [0, hidden_size] 的空输入。

    此时 rows 等于 0，
    CUDA Launcher 不应该启动 Kernel，
    而应该直接返回空输出。
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

    output = rmsnorm_cuda(
        x,
        weight,
    )

    assert output.shape == x.shape
    assert output.numel() == 0
    assert output.device.type == "cuda"
    assert output.dtype == torch.float32


def test_rejects_cpu_input() -> None:
    """
    x 位于 CPU 时，CUDA 接口应当拒绝执行。
    """

    x = torch.randn(
        2,
        128,
        dtype=torch.float32,
    )

    weight = torch.ones(
        128,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="x must be a CUDA tensor",
    ):
        rmsnorm_cuda(
            x,
            weight,
        )


def test_rejects_cpu_weight() -> None:
    """
    weight 位于 CPU 时，CUDA 接口应当拒绝执行。
    """

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        128,
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="weight must be a CUDA tensor",
    ):
        rmsnorm_cuda(
            x,
            weight,
        )


def test_rejects_fp16_input() -> None:
    """
    当前 Naive Kernel 只支持 FP32 输入。
    """

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float16,
    )

    weight = torch.ones(
        128,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="supports only float32 x",
    ):
        rmsnorm_cuda(
            x,
            weight,
        )


def test_rejects_fp16_weight() -> None:
    """
    当前 Naive Kernel 只支持 FP32 权重。
    """

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        128,
        device="cuda",
        dtype=torch.float16,
    )

    with pytest.raises(
        RuntimeError,
        match="supports only float32 weight",
    ):
        rmsnorm_cuda(
            x,
            weight,
        )


def test_rejects_mismatched_hidden_size() -> None:
    """
    weight.numel() 必须等于 x 的最后一个维度。
    """

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        64,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="must match weight.numel",
    ):
        rmsnorm_cuda(
            x,
            weight,
        )


def test_rejects_non_contiguous_input() -> None:
    """
    当前 CUDA Kernel 要求输入张量连续。
    """

    base = torch.randn(
        4,
        8,
        device="cuda",
        dtype=torch.float32,
    )

    # 转置后形状为 [8, 4]，
    # 但底层内存布局不再连续。
    x = base.transpose(0, 1)

    assert not x.is_contiguous()

    weight = torch.ones(
        4,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="x must be contiguous",
    ):
        rmsnorm_cuda(
            x,
            weight,
        )


def test_rejects_non_1d_weight() -> None:
    """
    weight 必须是一维张量。
    """

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        1,
        128,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="weight must be a 1-dimensional tensor",
    ):
        rmsnorm_cuda(
            x,
            weight,
        )


def test_rejects_scalar_input() -> None:
    """
    标量没有最后一个维度，因此不能执行 RMSNorm。
    """

    x = torch.tensor(
        1.0,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        1,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="at least one dimension",
    ):
        rmsnorm_cuda(
            x,
            weight,
        )


@pytest.mark.parametrize(
    "eps",
    [
        0.0,
        -1e-6,
    ],
)
def test_rejects_invalid_eps(
    eps: float,
) -> None:
    """
    eps 必须严格大于零。
    """

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.ones(
        128,
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="eps must be greater than zero",
    ):
        rmsnorm_cuda(
            x,
            weight,
            eps,
        )