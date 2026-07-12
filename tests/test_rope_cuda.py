"""
RoPE Naive CUDA Kernel 自动化测试。
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
from edge_kernelbench.rope_cuda import rope_cuda


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RoPE CUDA tests require a CUDA-capable device",
)


def assert_rope_cuda_matches_reference(
    shape: tuple[int, ...],
    use_1d_trig: bool = False,
) -> None:
    """
    创建随机 q/k/cos/sin，并比较 CUDA 与 PyTorch Reference。
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

    expected_q, expected_k = rope_reference(
        q,
        k,
        cos,
        sin,
    )

    actual_q, actual_k = rope_cuda(
        q,
        k,
        cos,
        sin,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_q,
        expected_q,
        rtol=1e-6,
        atol=1e-6,
    )

    torch.testing.assert_close(
        actual_k,
        expected_k,
        rtol=1e-6,
        atol=1e-6,
    )


def test_rope_cuda_known_values() -> None:
    """
    使用 90 度旋转验证 CUDA Kernel。
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

    expected_q, expected_k = rope_reference(
        q,
        k,
        cos,
        sin,
    )

    actual_q, actual_k = rope_cuda(
        q,
        k,
        cos,
        sin,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_q,
        expected_q,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        actual_k,
        expected_k,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 8),
        (4, 2, 8),
        (2, 4, 3, 8),
    ],
)
def test_rope_cuda_matches_reference_for_shapes(
    shape: tuple[int, ...],
) -> None:
    """
    验证常见 2D / 3D / 4D 形状。
    """

    assert_rope_cuda_matches_reference(
        shape
    )


def test_rope_cuda_supports_1d_cos_sin() -> None:
    """
    验证 [head_dim / 2] cos/sin 广播路径。
    """

    assert_rope_cuda_matches_reference(
        (2, 4, 3, 8),
        use_1d_trig=True,
    )


def test_rope_cuda_output_metadata() -> None:
    """
    输出应继承 q/k 的 shape、dtype、device 和连续布局。
    """

    q = torch.randn(
        2,
        4,
        3,
        8,
        device="cuda",
        dtype=torch.float32,
    )

    k = torch.randn_like(q)

    cos = torch.ones(
        4,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.zeros_like(cos)

    q_output, k_output = rope_cuda(
        q,
        k,
        cos,
        sin,
    )

    assert q_output.shape == q.shape
    assert k_output.shape == k.shape
    assert q_output.dtype == torch.float32
    assert k_output.dtype == torch.float32
    assert q_output.device.type == "cuda"
    assert k_output.device.type == "cuda"
    assert q_output.is_contiguous()
    assert k_output.is_contiguous()


def test_rope_cuda_empty_sequence() -> None:
    """
    空序列不应启动 CUDA Kernel。
    """

    q = torch.empty(
        0,
        2,
        8,
        device="cuda",
        dtype=torch.float32,
    )

    k = torch.empty_like(q)

    cos = torch.empty(
        0,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.empty_like(cos)

    q_output, k_output = rope_cuda(
        q,
        k,
        cos,
        sin,
    )

    assert q_output.shape == q.shape
    assert k_output.shape == k.shape
    assert q_output.numel() == 0
    assert k_output.numel() == 0


def test_rope_cuda_rejects_cpu_input() -> None:
    """
    q 位于 CPU 时应拒绝执行。
    """

    q = torch.randn(
        4,
        8,
        dtype=torch.float32,
    )

    k = torch.randn(
        4,
        8,
        device="cuda",
        dtype=torch.float32,
    )

    cos = torch.ones(
        4,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.zeros_like(cos)

    with pytest.raises(
        RuntimeError,
        match="q must be a CUDA tensor",
    ):
        rope_cuda(
            q,
            k,
            cos,
            sin,
        )


def test_rope_cuda_rejects_fp16_input() -> None:
    """
    当前 CUDA Kernel 只支持 FP32。
    """

    q = torch.randn(
        4,
        8,
        device="cuda",
        dtype=torch.float16,
    )

    k = torch.randn_like(q)

    cos = torch.ones(
        4,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.zeros_like(cos)

    with pytest.raises(
        RuntimeError,
        match="supports only float32 q",
    ):
        rope_cuda(
            q,
            k,
            cos,
            sin,
        )


def test_rope_cuda_rejects_non_contiguous_q() -> None:
    """
    当前 CUDA Kernel 要求 q contiguous。
    """

    base = torch.randn(
        8,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    q = base.transpose(
        0,
        1,
    )

    assert not q.is_contiguous()

    k = torch.randn(
        4,
        8,
        device="cuda",
        dtype=torch.float32,
    )

    cos = torch.ones(
        4,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.zeros_like(cos)

    with pytest.raises(
        RuntimeError,
        match="q must be contiguous",
    ):
        rope_cuda(
            q,
            k,
            cos,
            sin,
        )


def test_rope_cuda_rejects_odd_head_dim() -> None:
    """
    head_dim 必须为偶数。
    """

    q = torch.randn(
        4,
        7,
        device="cuda",
        dtype=torch.float32,
    )

    k = torch.randn_like(q)

    cos = torch.ones(
        4,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.zeros_like(cos)

    with pytest.raises(
        RuntimeError,
        match="must be even",
    ):
        rope_cuda(
            q,
            k,
            cos,
            sin,
        )


def test_rope_cuda_rejects_wrong_sequence_length() -> None:
    """
    cos.size(0) 必须匹配 sequence 维度。
    """

    q = torch.randn(
        2,
        4,
        3,
        8,
        device="cuda",
        dtype=torch.float32,
    )

    k = torch.randn_like(q)

    cos = torch.ones(
        5,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    sin = torch.zeros_like(cos)

    with pytest.raises(
        RuntimeError,
        match="sequence dimension",
    ):
        rope_cuda(
            q,
            k,
            cos,
            sin,
        )
