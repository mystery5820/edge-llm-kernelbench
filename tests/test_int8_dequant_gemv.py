"""
INT8 Dequant-GEMV PyTorch Reference 单元测试。
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


from edge_kernelbench.int8_dequant_gemv import (  # noqa: E402
    INT8DequantGEMVReference,
    int8_dequant_gemv_reference,
)


def available_devices() -> list[torch.device]:
    devices = [
        torch.device("cpu"),
    ]

    if torch.cuda.is_available():
        devices.append(
            torch.device("cuda")
        )

    return devices


DEVICES = available_devices()


def manual_int8_dequant_gemv(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    测试用直接公式实现。
    """

    compute_dtype = (
        torch.float32
        if x.dtype in (torch.float16, torch.bfloat16)
        else x.dtype
    )

    x_compute = x.to(
        dtype=compute_dtype
    )

    dequant_weight = (
        weight_int8.to(
            dtype=compute_dtype
        )
        * scale.to(
            dtype=compute_dtype
        )[:, None]
    )

    output = torch.matmul(
        x_compute,
        dequant_weight.transpose(
            0,
            1,
        ),
    )

    if bias is not None:
        output = (
            output
            + bias.to(
                dtype=compute_dtype
            )
        )

    return output.to(
        dtype=x.dtype
    )


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_known_values(
    device: torch.device,
) -> None:
    """
    使用固定小矩阵验证公式。
    """

    x = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
        ],
        device=device,
        dtype=torch.float32,
    )

    weight_int8 = torch.tensor(
        [
            [1, 2, 3],
            [-1, 0, 2],
        ],
        device=device,
        dtype=torch.int8,
    )

    scale = torch.tensor(
        [
            0.5,
            2.0,
        ],
        device=device,
        dtype=torch.float32,
    )

    bias = torch.tensor(
        [
            1.0,
            -1.0,
        ],
        device=device,
        dtype=torch.float32,
    )

    expected = manual_int8_dequant_gemv(
        x,
        weight_int8,
        scale,
        bias,
    )

    actual = int8_dequant_gemv_reference(
        x,
        weight_int8,
        scale,
        bias,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
@pytest.mark.parametrize(
    "shape",
    [
        (16,),
        (4, 16),
        (2, 3, 16),
    ],
)
def test_matches_manual_for_shapes(
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    """
    验证不同 batch 维度。
    """

    torch.manual_seed(2026)

    out_features = 12
    in_features = shape[-1]

    x = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
    )

    weight_int8 = torch.randint(
        low=-8,
        high=8,
        size=(
            out_features,
            in_features,
        ),
        device=device,
        dtype=torch.int8,
    )

    scale = torch.rand(
        out_features,
        device=device,
        dtype=torch.float32,
    )

    expected = manual_int8_dequant_gemv(
        x,
        weight_int8,
        scale,
    )

    actual = int8_dequant_gemv_reference(
        x,
        weight_int8,
        scale,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_fp16_cuda_output_dtype() -> None:
    """
    CUDA FP16 输入内部 FP32 计算，输出恢复 FP16。
    """

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is not available"
        )

    torch.manual_seed(2026)

    x = torch.randn(
        4,
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
        dtype=torch.float32,
    )

    output = int8_dequant_gemv_reference(
        x,
        weight_int8,
        scale,
    )

    assert output.shape == (
        4,
        12,
    )
    assert output.dtype == torch.float16
    assert output.device.type == "cuda"


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_module_matches_function(
    device: torch.device,
) -> None:
    """
    Module 封装应与函数式实现一致。
    """

    torch.manual_seed(2026)

    x = torch.randn(
        4,
        16,
        device=device,
    )

    weight_int8 = torch.randint(
        -8,
        8,
        (
            12,
            16,
        ),
        device=device,
        dtype=torch.int8,
    )

    scale = torch.rand(
        12,
        device=device,
    )

    bias = torch.randn(
        12,
        device=device,
    )

    module = INT8DequantGEMVReference(
        weight_int8,
        scale.clone(),
        bias.clone(),
    ).to(
        device=device
    )

    expected = int8_dequant_gemv_reference(
        x,
        weight_int8,
        module.scale,
        module.bias,
    )

    actual = module(
        x
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_backward_on_cuda() -> None:
    """
    Reference 应支持对 x、scale、bias 反向传播。
    """

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is not available"
        )

    torch.manual_seed(2026)

    x = torch.randn(
        4,
        16,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
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
        requires_grad=True,
    )

    bias = torch.randn(
        12,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    output = int8_dequant_gemv_reference(
        x,
        weight_int8,
        scale,
        bias,
    )

    loss = output.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert scale.grad is not None
    assert bias.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(scale.grad).all()
    assert torch.isfinite(bias.grad).all()


def test_rejects_non_float_x() -> None:
    x = torch.ones(
        4,
        dtype=torch.int64,
    )
    weight_int8 = torch.ones(
        2,
        4,
        dtype=torch.int8,
    )
    scale = torch.ones(
        2,
    )

    with pytest.raises(
        TypeError,
        match="x must be a floating-point tensor",
    ):
        int8_dequant_gemv_reference(
            x,
            weight_int8,
            scale,
        )


def test_rejects_non_int8_weight() -> None:
    x = torch.ones(
        4,
    )
    weight_int8 = torch.ones(
        2,
        4,
    )
    scale = torch.ones(
        2,
    )

    with pytest.raises(
        TypeError,
        match="weight_int8 must be torch.int8",
    ):
        int8_dequant_gemv_reference(
            x,
            weight_int8,
            scale,
        )


def test_rejects_mismatched_in_features() -> None:
    x = torch.ones(
        5,
    )
    weight_int8 = torch.ones(
        2,
        4,
        dtype=torch.int8,
    )
    scale = torch.ones(
        2,
    )

    with pytest.raises(
        ValueError,
        match="must match weight_int8.shape",
    ):
        int8_dequant_gemv_reference(
            x,
            weight_int8,
            scale,
        )


def test_rejects_mismatched_scale() -> None:
    x = torch.ones(
        4,
    )
    weight_int8 = torch.ones(
        2,
        4,
        dtype=torch.int8,
    )
    scale = torch.ones(
        3,
    )

    with pytest.raises(
        ValueError,
        match="scale.numel",
    ):
        int8_dequant_gemv_reference(
            x,
            weight_int8,
            scale,
        )


def test_rejects_device_mismatch() -> None:
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is not available"
        )

    x = torch.ones(
        4,
        device="cuda",
    )
    weight_int8 = torch.ones(
        2,
        4,
        dtype=torch.int8,
    )
    scale = torch.ones(
        2,
        device="cuda",
    )

    with pytest.raises(
        ValueError,
        match="weight_int8 and x must be on the same device",
    ):
        int8_dequant_gemv_reference(
            x,
            weight_int8,
            scale,
        )
