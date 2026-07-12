"""
RMSNorm PyTorch Reference 的单元测试。

本文件用于验证：

1. rmsnorm_reference() 的数学计算是否正确；
2. RMSNormReference 模块与函数式实现是否一致；
3. 输出的形状、数据类型和设备是否正确；
4. FP16 输入是否能够正常计算；
5. 反向传播是否能够生成梯度；
6. 输入参数检查是否能够正确抛出异常。

运行方式：

    python -m pytest tests/test_rmsnorm.py -v

由于项目代码位于：

    python/edge_kernelbench/

测试文件会在运行时把项目的 python 目录加入 sys.path，
从而可以直接导入 edge_kernelbench。
"""

# 导入 Python 标准库中的 sys。
#
# sys.path 保存了 Python 搜索模块的目录列表。
# 后面会把项目的 python/ 目录加入 sys.path。
import sys

# pathlib.Path 用于以更清晰、跨平台的方式处理文件路径。
from pathlib import Path

# pytest 是本项目使用的单元测试框架。
#
# 它负责：
#
# 1. 自动发现 test_ 开头的测试函数；
# 2. 执行测试；
# 3. 检查异常；
# 4. 输出测试通过或失败的信息。
import pytest

# 导入 PyTorch。
import torch


# ----------------------------------------------------------------------
# 配置项目代码导入路径
# ----------------------------------------------------------------------

# __file__ 表示当前测试文件：
#
#     tests/test_rmsnorm.py
#
# resolve() 会将它转换为绝对路径。
CURRENT_FILE = Path(__file__).resolve()

# parents[1] 表示当前文件向上两级的位置。
#
# 当前文件：
#
#     edge-llm-kernelbench/tests/test_rmsnorm.py
#
# parents[0]：
#
#     edge-llm-kernelbench/tests
#
# parents[1]：
#
#     edge-llm-kernelbench
#
# 因此 PROJECT_ROOT 就是项目根目录。
PROJECT_ROOT = CURRENT_FILE.parents[1]

# 项目的 Python 源代码位于：
#
#     edge-llm-kernelbench/python
PYTHON_SOURCE_DIR = PROJECT_ROOT / "python"

# 将 python/ 目录插入模块搜索路径的最前面。
#
# 完成后，Python 就能找到：
#
#     python/edge_kernelbench
#
# 从而允许执行：
#
#     from edge_kernelbench.rmsnorm import ...
sys.path.insert(
    0,
    str(PYTHON_SOURCE_DIR),
)


# ----------------------------------------------------------------------
# 导入需要测试的 RMSNorm 实现
# ----------------------------------------------------------------------

from edge_kernelbench.rmsnorm import (  # noqa: E402
    RMSNormReference,
    rmsnorm_reference,
)


# ----------------------------------------------------------------------
# 测试辅助函数
# ----------------------------------------------------------------------

def manual_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    使用最直接的数学公式计算 RMSNorm。

    这个函数只在测试中使用，作为被测实现的对照结果。

    数学过程：

        mean_square = mean(x²)

        rms = sqrt(mean_square + eps)

        output = x / rms * weight

    Args:
        x:
            输入张量，形状为 [..., hidden_size]。

        weight:
            缩放权重，形状为 [hidden_size]。

        eps:
            数值稳定常数。

    Returns:
        按照 RMSNorm 数学公式计算得到的结果。
    """

    # FP16 和 BF16 在平方、求和过程中误差较大，
    # 因此与正式实现相同，内部提升到 FP32。
    if x.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        compute_dtype = torch.float32
    else:
        compute_dtype = x.dtype

    # 转换到内部计算类型。
    x_compute = x.to(
        dtype=compute_dtype,
    )

    # 权重使用相同计算类型。
    weight_compute = weight.to(
        dtype=compute_dtype,
    )

    # 沿最后一个维度计算 x² 的平均值。
    mean_square = (
        x_compute
        .pow(2)
        .mean(
            dim=-1,
            keepdim=True,
        )
    )

    # 按数学定义计算：
    #
    #     x / sqrt(mean(x²) + eps)
    normalized = x_compute / torch.sqrt(
        mean_square + eps
    )

    # 应用缩放权重。
    output_compute = (
        normalized * weight_compute
    )

    # 输出数据类型恢复为输入类型。
    return output_compute.to(
        dtype=x.dtype,
    )


def available_devices() -> list[torch.device]:
    """
    返回当前环境中可以参与测试的设备列表。

    CPU 总是加入测试。

    如果 CUDA 可用，则同时测试 CUDA。
    """

    # 先加入 CPU。
    devices = [
        torch.device("cpu"),
    ]

    # Jetson 上 CUDA 正常时，再加入 cuda:0。
    if torch.cuda.is_available():
        devices.append(
            torch.device("cuda"),
        )

    return devices


# 在模块加载时生成设备列表。
#
# 当前 Jetson 环境中应该是：
#
#     [cpu, cuda]
DEVICES = available_devices()


# ----------------------------------------------------------------------
# 测试一：检查一组已知输入的数学结果
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_known_values(
    device: torch.device,
) -> None:
    """
    检查简单输入下的 RMSNorm 结果。

    输入：

        x = [1, 2, 3, 4]

    平方：

        x² = [1, 4, 9, 16]

    平方平均值：

        (1 + 4 + 9 + 16) / 4 = 7.5

    当 weight 全部为 1 时：

        output = x / sqrt(7.5 + eps)
    """

    # 设置数值稳定常数。
    eps = 1e-6

    # 创建一行简单输入。
    x = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    # weight 全部初始化为 1。
    weight = torch.ones(
        4,
        dtype=torch.float32,
        device=device,
    )

    # 调用项目中的 RMSNorm 参考实现。
    actual = rmsnorm_reference(
        x=x,
        weight=weight,
        eps=eps,
    )

    # 根据已知数学结果构造期望输出。
    expected = x / torch.sqrt(
        torch.tensor(
            7.5 + eps,
            dtype=torch.float32,
            device=device,
        )
    )

    # 检查实际结果和期望结果是否足够接近。
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


# ----------------------------------------------------------------------
# 测试二：随机输入与直接数学公式是否一致
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_reference_matches_manual_formula(
    device: torch.device,
) -> None:
    """
    使用随机张量验证 rmsnorm_reference()。

    输入形状设置为：

        [batch_size, sequence_length, hidden_size]

        [2, 3, 16]

    这可以验证实现确实沿最后一个维度进行归一化。
    """

    # 固定随机种子。
    #
    # 每次运行测试时会生成相同的随机数据，
    # 便于复现问题。
    torch.manual_seed(2026)

    # 创建三维输入张量。
    x = torch.randn(
        2,
        3,
        16,
        dtype=torch.float32,
        device=device,
    )

    # 创建随机缩放权重。
    weight = torch.randn(
        16,
        dtype=torch.float32,
        device=device,
    )

    # 设置 eps。
    eps = 1e-6

    # 调用被测试的实现。
    actual = rmsnorm_reference(
        x=x,
        weight=weight,
        eps=eps,
    )

    # 调用测试文件中的直接数学公式。
    expected = manual_rmsnorm(
        x=x,
        weight=weight,
        eps=eps,
    )

    # 比较两种实现。
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


# ----------------------------------------------------------------------
# 测试三：函数式实现和 nn.Module 实现是否一致
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_function_and_module_match(
    device: torch.device,
) -> None:
    """
    验证：

        rmsnorm_reference()

    与：

        RMSNormReference

    在相同输入和相同 weight 下得到相同结果。
    """

    # 固定随机种子。
    torch.manual_seed(2026)

    # 定义 hidden_size。
    hidden_size = 32

    # 创建输入。
    x = torch.randn(
        4,
        hidden_size,
        dtype=torch.float32,
        device=device,
    )

    # 创建 RMSNorm 模块并移动到目标设备。
    layer = RMSNormReference(
        hidden_size=hidden_size,
        eps=1e-6,
    ).to(device)

    # 为模块的 weight 设置随机数据。
    #
    # no_grad() 表示这次赋值不记录到计算图中。
    with torch.no_grad():
        layer.weight.copy_(
            torch.randn(
                hidden_size,
                dtype=torch.float32,
                device=device,
            )
        )

    # 函数式实现结果。
    function_output = rmsnorm_reference(
        x=x,
        weight=layer.weight,
        eps=layer.eps,
    )

    # nn.Module 实现结果。
    module_output = layer(x)

    # 两者应该完全一致或仅存在极小浮点误差。
    torch.testing.assert_close(
        function_output,
        module_output,
        rtol=1e-5,
        atol=1e-6,
    )


# ----------------------------------------------------------------------
# 测试四：输出形状、类型和设备是否保持一致
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "device",
    DEVICES,
    ids=lambda device: device.type,
)
def test_output_metadata(
    device: torch.device,
) -> None:
    """
    检查 RMSNorm 输出的：

    1. shape；
    2. dtype；
    3. device。

    它们都应该与输入 x 保持一致。
    """

    # 创建输入。
    x = torch.randn(
        2,
        5,
        64,
        dtype=torch.float32,
        device=device,
    )

    # 创建权重。
    weight = torch.ones(
        64,
        dtype=torch.float32,
        device=device,
    )

    # 执行 RMSNorm。
    output = rmsnorm_reference(
        x=x,
        weight=weight,
    )

    # 检查输出形状。
    assert output.shape == x.shape

    # 检查输出数据类型。
    assert output.dtype == x.dtype

    # 检查输出设备。
    assert output.device == x.device


# ----------------------------------------------------------------------
# 测试五：CUDA FP16 输入是否正常
# ----------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the FP16 test",
)
def test_fp16_cuda() -> None:
    """
    验证 FP16 CUDA 输入。

    实现内部会使用 FP32 完成平方、平均值和开方，
    最后再把输出转换回 FP16。
    """

    # 固定随机种子。
    torch.manual_seed(2026)

    # 创建 FP16 CUDA 输入。
    x = torch.randn(
        2,
        128,
        dtype=torch.float16,
        device="cuda",
    )

    # 创建 FP16 权重。
    weight = torch.randn(
        128,
        dtype=torch.float16,
        device="cuda",
    )

    # 项目实现结果。
    actual = rmsnorm_reference(
        x=x,
        weight=weight,
        eps=1e-6,
    )

    # 测试数学公式结果。
    expected = manual_rmsnorm(
        x=x,
        weight=weight,
        eps=1e-6,
    )

    # 输出仍然应该是 FP16。
    assert actual.dtype == torch.float16

    # 输出仍然应该位于 CUDA。
    assert actual.device.type == "cuda"

    # FP16 的误差阈值需要比 FP32 宽松。
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-3,
        atol=1e-3,
    )


# ----------------------------------------------------------------------
# 测试六：反向传播是否正常
# ----------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for this gradient test",
)
def test_backward_on_cuda() -> None:
    """
    验证 RMSNorm 是否支持 PyTorch 自动求导。

    运行 backward() 后：

    1. 输入 x 应该得到梯度；
    2. 模块参数 weight 应该得到梯度；
    3. 梯度中不能出现 NaN 或 Inf。
    """

    # 固定随机种子。
    torch.manual_seed(2026)

    # 创建 requires_grad=True 的输入。
    #
    # 这表示 PyTorch 需要记录与 x 有关的计算，
    # 以便后续求出 x.grad。
    x = torch.randn(
        2,
        64,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )

    # 创建 RMSNorm 模块并移动到 CUDA。
    layer = RMSNormReference(
        hidden_size=64,
    ).cuda()

    # 执行前向计算。
    output = layer(x)

    # 构造一个标量损失。
    #
    # backward() 默认要求输出是标量，
    # 因此这里对所有输出求和。
    loss = output.sum()

    # 执行反向传播。
    loss.backward()

    # 输入 x 应该已经生成梯度。
    assert x.grad is not None

    # 模块参数 weight 也应该生成梯度。
    assert layer.weight.grad is not None

    # 检查输入梯度全部是有限数。
    assert torch.isfinite(
        x.grad
    ).all()

    # 检查权重梯度全部是有限数。
    assert torch.isfinite(
        layer.weight.grad
    ).all()


# ----------------------------------------------------------------------
# 测试七：整数输入应该抛出 TypeError
# ----------------------------------------------------------------------

def test_rejects_integer_input() -> None:
    """
    RMSNorm 不能直接处理整数输入。
    """

    # 创建整数输入。
    x = torch.tensor(
        [
            [
                1,
                2,
                3,
                4,
            ]
        ],
        dtype=torch.int64,
    )

    # 创建浮点权重。
    weight = torch.ones(
        4,
        dtype=torch.float32,
    )

    # pytest.raises() 用于检查代码是否抛出指定异常。
    with pytest.raises(
        TypeError,
        match="x must be a floating-point tensor",
    ):
        rmsnorm_reference(
            x=x,
            weight=weight,
        )


# ----------------------------------------------------------------------
# 测试八：weight 不是一维时应该报错
# ----------------------------------------------------------------------

def test_rejects_non_1d_weight() -> None:
    """
    weight 必须是 [hidden_size] 的一维张量。
    """

    # 正常输入。
    x = torch.randn(
        2,
        4,
    )

    # 错误的二维 weight。
    weight = torch.ones(
        1,
        4,
    )

    # 应该抛出 ValueError。
    with pytest.raises(
        ValueError,
        match="weight must be a 1-dimensional tensor",
    ):
        rmsnorm_reference(
            x=x,
            weight=weight,
        )


# ----------------------------------------------------------------------
# 测试九：hidden_size 不匹配时应该报错
# ----------------------------------------------------------------------

def test_rejects_mismatched_hidden_size() -> None:
    """
    x 的最后一维必须与 weight 元素数量一致。
    """

    # 输入的 hidden_size 为 8。
    x = torch.randn(
        2,
        8,
    )

    # weight 只有 4 个元素。
    weight = torch.ones(
        4,
    )

    # 应该抛出 ValueError。
    with pytest.raises(
        ValueError,
        match="The last dimension of x must match weight.numel",
    ):
        rmsnorm_reference(
            x=x,
            weight=weight,
        )


# ----------------------------------------------------------------------
# 测试十：CPU 和 CUDA 混用时应该报错
# ----------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the mixed-device test",
)
def test_rejects_different_devices() -> None:
    """
    x 和 weight 必须位于同一个设备。
    """

    # x 位于 CUDA。
    x = torch.randn(
        2,
        8,
        device="cuda",
    )

    # weight 位于 CPU。
    weight = torch.ones(
        8,
        device="cpu",
    )

    # 应该在正式计算前主动报错。
    with pytest.raises(
        ValueError,
        match="x and weight must be on the same device",
    ):
        rmsnorm_reference(
            x=x,
            weight=weight,
        )


# ----------------------------------------------------------------------
# 测试十一：非法 eps 应该报错
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_eps",
    [
        0.0,
        -1e-6,
    ],
)
def test_rejects_invalid_eps(
    invalid_eps: float,
) -> None:
    """
    eps 必须严格大于 0。
    """

    # 创建正常输入。
    x = torch.randn(
        2,
        8,
    )

    # 创建正常权重。
    weight = torch.ones(
        8,
    )

    # eps 为 0 或负数时应该报错。
    with pytest.raises(
        ValueError,
        match="eps must be greater than zero",
    ):
        rmsnorm_reference(
            x=x,
            weight=weight,
            eps=invalid_eps,
        )


# ----------------------------------------------------------------------
# 测试十二：非法 hidden_size 应该报错
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_hidden_size",
    [
        0,
        -1,
        -128,
    ],
)
def test_module_rejects_invalid_hidden_size(
    invalid_hidden_size: int,
) -> None:
    """
    RMSNormReference 的 hidden_size 必须大于 0。
    """

    # 创建模块时应该立即抛出 ValueError。
    with pytest.raises(
        ValueError,
        match="hidden_size must be greater than zero",
    ):
        RMSNormReference(
            hidden_size=invalid_hidden_size,
        )