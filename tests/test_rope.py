"""
RoPE PyTorch Reference 的单元测试。

本文件用于验证：

1. apply_rope_reference() 的数学公式是否正确；
2. rope_reference() 是否能同时处理 q / k；
3. RoPEReference 模块与函数式实现是否一致；
4. 常见 2D / 3D / 4D 输入形状是否正确；
5. FP16 CUDA 内部 FP32 计算是否可用；
6. 反向传播是否可用；
7. 输入检查是否能正确抛出异常。
"""

from itertools import product
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


from edge_kernelbench.rope import (  # noqa: E402
    RoPEReference,
    apply_rope_reference,
    rope_reference,
)


def available_devices() -> list[torch.device]:
    """
    返回当前可用于 RoPE reference 测试的设备。
    """

    devices = [
        torch.device("cpu"),
    ]

    if torch.cuda.is_available():
        devices.append(
            torch.device("cuda"),
        )

    return devices


DEVICES = available_devices()


# ---------------------------------------------------------------------------
# 测试辅助函数
# ---------------------------------------------------------------------------


def manual_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    使用直接循环实现 RoPE，用作测试对照。

    该函数只覆盖测试中使用的 cos / sin 形状：

    - [head_dim / 2]
    - [seq_len, head_dim / 2]
    """

    output = torch.empty_like(x)
    half_dim = x.shape[-1] // 2

    prefix_shape = x.shape[:-1]

    if cos.ndim == 1:
        for prefix in product(
            *[
                range(size)
                for size in prefix_shape
            ]
        ):
            for pair_id in range(half_dim):
                even_index = prefix + (pair_id * 2,)
                odd_index = prefix + (pair_id * 2 + 1,)

                even_value = x[even_index]
                odd_value = x[odd_index]

                output[even_index] = (
                    even_value
                    * cos[pair_id]
                    - odd_value
                    * sin[pair_id]
                )

                output[odd_index] = (
                    even_value
                    * sin[pair_id]
                    + odd_value
                    * cos[pair_id]
                )

        return output

    if cos.ndim != 2:
        raise ValueError(
            "manual_apply_rope supports only 1D or 2D cos/sin"
        )

    sequence_axis = (
        0
        if x.ndim == 2
        else x.ndim - 3
    )

    for prefix in product(
        *[
            range(size)
            for size in prefix_shape
        ]
    ):
        sequence_index = prefix[sequence_axis]

        for pair_id in range(half_dim):
            even_index = prefix + (pair_id * 2,)
            odd_index = prefix + (pair_id * 2 + 1,)

            even_value = x[even_index]
            odd_value = x[odd_index]

            output[even_index] = (
                even_value
                * cos[sequence_index, pair_id]
                - odd_value
                * sin[sequence_index, pair_id]
            )

            output[odd_index] = (
                even_value
                * sin[sequence_index, pair_id]
                + odd_value
                * cos[sequence_index, pair_id]
            )

    return output


# ---------------------------------------------------------------------------
# 固定值测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_apply_rope_known_values(
    device: torch.device,
) -> None:
    """
    使用 90 度旋转验证 RoPE 公式。
    """

    x = torch.tensor(
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
        device=device,
        dtype=torch.float32,
    )

    cos = torch.zeros(
        1,
        2,
        device=device,
        dtype=torch.float32,
    )

    sin = torch.ones(
        1,
        2,
        device=device,
        dtype=torch.float32,
    )

    actual = apply_rope_reference(
        x,
        cos,
        sin,
    )

    expected = torch.tensor(
        [
            [
                [
                    -2.0,
                    1.0,
                    -4.0,
                    3.0,
                ]
            ]
        ],
        device=device,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# 多形状测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 8),
        (4, 2, 8),
        (2, 4, 3, 8),
    ],
)
def test_apply_rope_matches_manual_for_shapes(
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    """
    验证 2D / 3D / 4D 输入的 RoPE 结果。
    """

    torch.manual_seed(2026)

    x = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
    )

    seq_len = (
        shape[0]
        if len(shape) == 2
        else shape[-3]
    )

    half_dim = shape[-1] // 2

    angles = torch.randn(
        seq_len,
        half_dim,
        device=device,
        dtype=torch.float32,
    )

    cos = torch.cos(
        angles
    )

    sin = torch.sin(
        angles
    )

    expected = manual_apply_rope(
        x,
        cos,
        sin,
    )

    actual = apply_rope_reference(
        x,
        cos,
        sin,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_apply_rope_supports_1d_cos_sin(
    device: torch.device,
) -> None:
    """
    验证 [head_dim / 2] 形状的 cos / sin 可以广播到所有位置。
    """

    torch.manual_seed(2026)

    x = torch.randn(
        2,
        3,
        4,
        8,
        device=device,
        dtype=torch.float32,
    )

    angles = torch.randn(
        4,
        device=device,
        dtype=torch.float32,
    )

    cos = torch.cos(
        angles
    )

    sin = torch.sin(
        angles
    )

    expected = manual_apply_rope(
        x,
        cos,
        sin,
    )

    actual = apply_rope_reference(
        x,
        cos,
        sin,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# q/k 和 Module 测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_rope_reference_handles_q_and_k(
    device: torch.device,
) -> None:
    """
    rope_reference() 应同时返回 q 和 k 的 RoPE 结果。
    """

    torch.manual_seed(2026)

    q = torch.randn(
        2,
        4,
        3,
        8,
        device=device,
        dtype=torch.float32,
    )

    k = torch.randn_like(q)

    angles = torch.randn(
        4,
        4,
        device=device,
        dtype=torch.float32,
    )

    cos = torch.cos(
        angles
    )

    sin = torch.sin(
        angles
    )

    q_actual, k_actual = rope_reference(
        q,
        k,
        cos,
        sin,
    )

    q_expected = apply_rope_reference(
        q,
        cos,
        sin,
    )

    k_expected = apply_rope_reference(
        k,
        cos,
        sin,
    )

    torch.testing.assert_close(
        q_actual,
        q_expected,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        k_actual,
        k_expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_rope_module_matches_function(
    device: torch.device,
) -> None:
    """
    RoPEReference 模块输出应与函数式实现一致。
    """

    torch.manual_seed(2026)

    q = torch.randn(
        4,
        2,
        8,
        device=device,
        dtype=torch.float32,
    )

    k = torch.randn_like(q)

    angles = torch.randn(
        4,
        4,
        device=device,
        dtype=torch.float32,
    )

    cos = torch.cos(
        angles
    )

    sin = torch.sin(
        angles
    )

    module = RoPEReference()

    module_q, module_k = module(
        q,
        k,
        cos,
        sin,
    )

    function_q, function_k = rope_reference(
        q,
        k,
        cos,
        sin,
    )

    torch.testing.assert_close(
        module_q,
        function_q,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        module_k,
        function_k,
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# dtype、metadata 和 autograd
# ---------------------------------------------------------------------------


def test_apply_rope_fp16_cuda() -> None:
    """
    验证 CUDA FP16 输入可以执行，并保持输出 dtype。
    """

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is not available"
        )

    torch.manual_seed(2026)

    x = torch.randn(
        2,
        4,
        3,
        8,
        device="cuda",
        dtype=torch.float16,
    )

    angles = torch.randn(
        4,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    cos = torch.cos(
        angles
    )

    sin = torch.sin(
        angles
    )

    output = apply_rope_reference(
        x,
        cos,
        sin,
    )

    assert output.shape == x.shape
    assert output.dtype == torch.float16
    assert output.device == x.device


@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_apply_rope_output_metadata(
    device: torch.device,
) -> None:
    """
    输出应继承输入 shape、dtype 和 device。
    """

    x = torch.randn(
        2,
        4,
        3,
        8,
        device=device,
        dtype=torch.float32,
    )

    cos = torch.ones(
        4,
        4,
        device=device,
        dtype=torch.float32,
    )

    sin = torch.zeros(
        4,
        4,
        device=device,
        dtype=torch.float32,
    )

    output = apply_rope_reference(
        x,
        cos,
        sin,
    )

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


def test_apply_rope_backward_on_cuda() -> None:
    """
    验证 RoPE reference 支持反向传播。
    """

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is not available"
        )

    torch.manual_seed(2026)

    x = torch.randn(
        2,
        4,
        3,
        8,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    angles = torch.randn(
        4,
        4,
        device="cuda",
        dtype=torch.float32,
    )

    cos = torch.cos(
        angles
    )

    sin = torch.sin(
        angles
    )

    output = apply_rope_reference(
        x,
        cos,
        sin,
    )

    loss = output.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 输入检查
# ---------------------------------------------------------------------------


def test_rejects_integer_input() -> None:
    """
    x 必须是浮点张量。
    """

    x = torch.ones(
        4,
        8,
        dtype=torch.int64,
    )

    cos = torch.ones(
        4,
        4,
    )

    sin = torch.zeros(
        4,
        4,
    )

    with pytest.raises(
        TypeError,
        match="x must be a floating-point tensor",
    ):
        apply_rope_reference(
            x,
            cos,
            sin,
        )


def test_rejects_odd_head_dim() -> None:
    """
    RoPE 要求最后一维为偶数。
    """

    x = torch.randn(
        4,
        7,
    )

    cos = torch.ones(
        4,
        4,
    )

    sin = torch.zeros(
        4,
        4,
    )

    with pytest.raises(
        ValueError,
        match="must be even",
    ):
        apply_rope_reference(
            x,
            cos,
            sin,
        )


def test_rejects_wrong_cos_hidden_size() -> None:
    """
    cos 最后一维必须等于 head_dim / 2。
    """

    x = torch.randn(
        4,
        8,
    )

    cos = torch.ones(
        4,
        3,
    )

    sin = torch.zeros(
        4,
        3,
    )

    with pytest.raises(
        ValueError,
        match="head_dim / 2",
    ):
        apply_rope_reference(
            x,
            cos,
            sin,
        )


def test_rejects_wrong_sequence_length() -> None:
    """
    cos 的 seq_len 必须匹配 x 的 sequence 维度。
    """

    x = torch.randn(
        2,
        4,
        3,
        8,
    )

    cos = torch.ones(
        5,
        4,
    )

    sin = torch.zeros(
        5,
        4,
    )

    with pytest.raises(
        ValueError,
        match="sequence dimension",
    ):
        apply_rope_reference(
            x,
            cos,
            sin,
        )


def test_rejects_mismatched_q_and_k_shape() -> None:
    """
    q 和 k 必须具有相同 shape。
    """

    q = torch.randn(
        4,
        2,
        8,
    )

    k = torch.randn(
        4,
        3,
        8,
    )

    cos = torch.ones(
        4,
        4,
    )

    sin = torch.zeros(
        4,
        4,
    )

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        rope_reference(
            q,
            k,
            cos,
            sin,
        )


def test_rejects_device_mismatch() -> None:
    """
    x、cos、sin 必须位于同一设备。
    """

    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is not available"
        )

    x = torch.randn(
        4,
        8,
        device="cuda",
    )

    cos = torch.ones(
        4,
        4,
        device="cpu",
    )

    sin = torch.zeros(
        4,
        4,
        device="cuda",
    )

    with pytest.raises(
        ValueError,
        match="cos and x must be on the same device",
    ):
        apply_rope_reference(
            x,
            cos,
            sin,
        )
