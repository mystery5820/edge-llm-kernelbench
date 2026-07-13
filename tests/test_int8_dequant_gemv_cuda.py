"""
INT8 Dequant-GEMV Naive CUDA Kernel 自动化测试。
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


from edge_kernelbench.int8_dequant_gemv import int8_dequant_gemv_reference
from edge_kernelbench.int8_dequant_gemv_cuda import (
    int8_dequant_gemv_cuda,
    int8_dequant_gemv_cuda_tiled,
    int8_dequant_gemv_cuda_vec4,
    int8_dequant_gemv_cuda_warp,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="INT8 Dequant-GEMV CUDA tests require a CUDA-capable device",
)


def assert_cuda_matches_reference(
    shape: tuple[int, ...],
    out_features: int,
    use_bias: bool = False,
) -> None:
    """
    创建随机输入，并比较 CUDA 与 PyTorch Reference。
    """

    torch.manual_seed(2026)

    in_features = shape[-1]

    x = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float32,
    )

    weight_int8 = torch.randint(
        -8,
        8,
        (
            out_features,
            in_features,
        ),
        device="cuda",
        dtype=torch.int8,
    )

    scale = torch.rand(
        out_features,
        device="cuda",
        dtype=torch.float32,
    )

    bias = None

    if use_bias:
        bias = torch.randn(
            out_features,
            device="cuda",
            dtype=torch.float32,
        )

    expected = int8_dequant_gemv_reference(
        x,
        weight_int8,
        scale,
        bias,
    )

    actual = int8_dequant_gemv_cuda(
        x,
        weight_int8,
        scale,
        bias,
    )

    warp_actual = int8_dequant_gemv_cuda_warp(
        x,
        weight_int8,
        scale,
        bias,
    )

    tiled_actual = int8_dequant_gemv_cuda_tiled(
        x,
        weight_int8,
        scale,
        bias,
    )

    vec4_actual = int8_dequant_gemv_cuda_vec4(
        x,
        weight_int8,
        scale,
        bias,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        warp_actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        warp_actual,
        actual,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        tiled_actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        tiled_actual,
        warp_actual,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        vec4_actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        vec4_actual,
        warp_actual,
        rtol=1e-4,
        atol=1e-4,
    )


def test_cuda_known_values() -> None:
    """
    使用固定小输入验证 CUDA Kernel。
    """

    x = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )

    weight_int8 = torch.tensor(
        [
            [1, 2, 3],
            [-1, 0, 2],
        ],
        device="cuda",
        dtype=torch.int8,
    )

    scale = torch.tensor(
        [
            0.5,
            2.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )

    bias = torch.tensor(
        [
            1.0,
            -1.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )

    expected = int8_dequant_gemv_reference(
        x,
        weight_int8,
        scale,
        bias,
    )

    actual = int8_dequant_gemv_cuda(
        x,
        weight_int8,
        scale,
        bias,
    )

    warp_actual = int8_dequant_gemv_cuda_warp(
        x,
        weight_int8,
        scale,
        bias,
    )

    tiled_actual = int8_dequant_gemv_cuda_tiled(
        x,
        weight_int8,
        scale,
        bias,
    )

    vec4_actual = int8_dequant_gemv_cuda_vec4(
        x,
        weight_int8,
        scale,
        bias,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        warp_actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        tiled_actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        vec4_actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (16,),
        (4, 16),
        (4, 18),
        (2, 3, 16),
    ],
)
def test_cuda_matches_reference_for_shapes(
    shape: tuple[int, ...],
) -> None:
    assert_cuda_matches_reference(
        shape,
        out_features=12,
    )


def test_cuda_matches_reference_with_bias() -> None:
    assert_cuda_matches_reference(
        (4, 16),
        out_features=12,
        use_bias=True,
    )


def test_cuda_output_metadata() -> None:
    """
    输出 shape、dtype、device 应正确。
    """

    x = torch.randn(
        2,
        3,
        16,
        device="cuda",
        dtype=torch.float32,
    )

    weight_int8 = torch.randint(
        -8,
        8,
        (
            12,
            16,
        ),
        device="cuda",
        dtype=torch.int8,
    )

    scale = torch.rand(
        12,
        device="cuda",
        dtype=torch.float32,
    )

    output = int8_dequant_gemv_cuda(
        x,
        weight_int8,
        scale,
    )

    assert output.shape == (
        2,
        3,
        12,
    )
    assert output.dtype == torch.float32
    assert output.device.type == "cuda"
    assert output.is_contiguous()


def test_cuda_empty_rows() -> None:
    """
    rows=0 时不应启动 kernel。
    """

    x = torch.empty(
        0,
        16,
        device="cuda",
        dtype=torch.float32,
    )

    weight_int8 = torch.empty(
        12,
        16,
        device="cuda",
        dtype=torch.int8,
    )

    scale = torch.empty(
        12,
        device="cuda",
        dtype=torch.float32,
    )

    output = int8_dequant_gemv_cuda(
        x,
        weight_int8,
        scale,
    )

    assert output.shape == (
        0,
        12,
    )
    assert output.numel() == 0


def test_cuda_rejects_cpu_input() -> None:
    x = torch.randn(
        16,
        dtype=torch.float32,
    )
    weight_int8 = torch.randint(
        -8,
        8,
        (
            12,
            16,
        ),
        device="cuda",
        dtype=torch.int8,
    )
    scale = torch.rand(
        12,
        device="cuda",
    )

    with pytest.raises(
        RuntimeError,
        match="x must be a CUDA tensor",
    ):
        int8_dequant_gemv_cuda(
            x,
            weight_int8,
            scale,
        )


def test_cuda_rejects_fp16_input() -> None:
    x = torch.randn(
        16,
        device="cuda",
        dtype=torch.float16,
    )
    weight_int8 = torch.randint(
        -8,
        8,
        (
            12,
            16,
        ),
        device="cuda",
        dtype=torch.int8,
    )
    scale = torch.rand(
        12,
        device="cuda",
    )

    with pytest.raises(
        RuntimeError,
        match="supports only float32 x",
    ):
        int8_dequant_gemv_cuda(
            x,
            weight_int8,
            scale,
        )


def test_cuda_rejects_non_contiguous_input() -> None:
    base = torch.randn(
        16,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    x = base.transpose(
        0,
        1,
    )

    assert not x.is_contiguous()

    weight_int8 = torch.randint(
        -8,
        8,
        (
            12,
            16,
        ),
        device="cuda",
        dtype=torch.int8,
    )
    scale = torch.rand(
        12,
        device="cuda",
    )

    with pytest.raises(
        RuntimeError,
        match="x must be contiguous",
    ):
        int8_dequant_gemv_cuda(
            x,
            weight_int8,
            scale,
        )
