"""
RoPE float4 CUDA Kernel 自动化测试。
"""

from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIRECTORY = PROJECT_ROOT / "python"

if str(PYTHON_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIRECTORY),
    )


from edge_kernelbench.rope import rope_reference
from edge_kernelbench.rope_cuda import (
    rope_cuda,
    rope_cuda_float4,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RoPE float4 CUDA tests require a CUDA-capable device",
)


def assert_float4_matches_reference_and_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """
    比较 PyTorch Reference、CUDA Naive 和 CUDA Float4。
    """

    reference_q, reference_k = rope_reference(
        q,
        k,
        cos,
        sin,
    )

    naive_q, naive_k = rope_cuda(
        q,
        k,
        cos,
        sin,
    )

    float4_q, float4_k = rope_cuda_float4(
        q,
        k,
        cos,
        sin,
    )

    torch.cuda.synchronize()

    for actual, expected in [
        (float4_q, reference_q),
        (float4_k, reference_k),
        (float4_q, naive_q),
        (float4_k, naive_k),
    ]:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )


def make_inputs(
    shape: tuple[int, ...],
    use_1d_trig: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    创建 RoPE CUDA 测试输入。
    """

    torch.manual_seed(2026)

    q = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float32,
    )

    k = torch.randn_like(q)

    half_dim = shape[-1] // 2

    if use_1d_trig:
        angles = torch.randn(
            half_dim,
            device="cuda",
            dtype=torch.float32,
        )
    else:
        seq_len = (
            shape[0]
            if len(shape) == 2
            else shape[-3]
        )

        angles = torch.randn(
            seq_len,
            half_dim,
            device="cuda",
            dtype=torch.float32,
        )

    cos = torch.cos(
        angles
    ).contiguous()

    sin = torch.sin(
        angles
    ).contiguous()

    return (
        q,
        k,
        cos,
        sin,
    )


def test_rope_float4_known_values() -> None:
    """
    使用 90 度旋转验证 float4 CUDA Kernel。
    """

    q = torch.tensor(
        [
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            ]
        ],
        device="cuda",
        dtype=torch.float32,
    )

    k = q + 10.0

    cos = torch.zeros(
        1,
        2,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.ones(
        1,
        2,
        device="cuda",
        dtype=torch.float32,
    )

    assert_float4_matches_reference_and_naive(
        q,
        k,
        cos,
        sin,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 8),
        (4, 2, 64),
        (2, 4, 3, 64),
        (2, 4, 3, 128),
    ],
)
def test_rope_float4_matches_reference_for_shapes(
    shape: tuple[int, ...],
) -> None:
    """
    验证常见向量化形状。
    """

    q, k, cos, sin = make_inputs(
        shape
    )

    assert_float4_matches_reference_and_naive(
        q,
        k,
        cos,
        sin,
    )


def test_rope_float4_supports_1d_cos_sin() -> None:
    """
    验证 1D cos/sin 广播路径。
    """

    q, k, cos, sin = make_inputs(
        (2, 4, 3, 64),
        use_1d_trig=True,
    )

    assert_float4_matches_reference_and_naive(
        q,
        k,
        cos,
        sin,
    )


def test_rope_float4_head_dim_not_multiple_of_four_falls_back() -> None:
    """
    head_dim 为偶数但不能被 4 整除时应走标量回退。
    """

    q, k, cos, sin = make_inputs(
        (2, 4, 3, 10)
    )

    assert_float4_matches_reference_and_naive(
        q,
        k,
        cos,
        sin,
    )


def test_rope_float4_unaligned_contiguous_input_falls_back() -> None:
    """
    storage_offset=1 的连续张量首地址不满足 16 字节对齐，应安全回退。
    """

    torch.manual_seed(2026)

    shape = (
        2,
        4,
        3,
        64,
    )

    numel = 1
    for size in shape:
        numel *= size

    q_base = torch.randn(
        numel + 1,
        device="cuda",
        dtype=torch.float32,
    )

    k_base = torch.randn(
        numel + 1,
        device="cuda",
        dtype=torch.float32,
    )

    q = q_base[1:].view(
        shape
    )

    k = k_base[1:].view(
        shape
    )

    assert q.is_contiguous()
    assert k.is_contiguous()
    assert q.storage_offset() == 1
    assert k.storage_offset() == 1

    angles = torch.randn(
        shape[-3],
        shape[-1] // 2,
        device="cuda",
        dtype=torch.float32,
    )

    cos = torch.cos(
        angles
    ).contiguous()

    sin = torch.sin(
        angles
    ).contiguous()

    assert_float4_matches_reference_and_naive(
        q,
        k,
        cos,
        sin,
    )


def test_rope_float4_output_metadata() -> None:
    """
    输出应继承输入 metadata。
    """

    q, k, cos, sin = make_inputs(
        (2, 4, 3, 64)
    )

    q_output, k_output = rope_cuda_float4(
        q,
        k,
        cos,
        sin,
    )

    assert q_output.shape == q.shape
    assert k_output.shape == k.shape
    assert q_output.dtype == q.dtype
    assert k_output.dtype == k.dtype
    assert q_output.device == q.device
    assert k_output.device == k.device
    assert q_output.is_contiguous()
    assert k_output.is_contiguous()


def test_rope_float4_empty_sequence() -> None:
    """
    空序列应安全返回。
    """

    q = torch.empty(
        0,
        2,
        64,
        device="cuda",
        dtype=torch.float32,
    )

    k = torch.empty_like(q)

    cos = torch.empty(
        0,
        32,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.empty_like(cos)

    q_output, k_output = rope_cuda_float4(
        q,
        k,
        cos,
        sin,
    )

    assert q_output.shape == q.shape
    assert k_output.shape == k.shape
    assert q_output.numel() == 0
    assert k_output.numel() == 0
