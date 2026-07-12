"""
RMSNorm 的 PyTorch 参考实现。

这个文件暂时不包含自定义 CUDA Kernel，而是使用标准 PyTorch 运算实现 RMSNorm。

它的主要作用有两个：

1. 作为 RMSNorm 数学逻辑的标准实现；
2. 后续用于验证 CUDA Kernel 的计算结果是否正确。

RMSNorm 的核心公式为：

    mean_square = mean(x²)

    inverse_rms = 1 / sqrt(mean_square + eps)

    output = x * inverse_rms * weight

其中归一化操作沿输入张量的最后一个维度进行。
"""

# 允许在类型注解中更灵活地引用尚未完全解析的类型。
# 在 Python 3.10 中，这能减少部分类型注解兼容问题。
from __future__ import annotations

# 导入 PyTorch。
# torch.Tensor、torch.float32、torch.rsqrt 等功能都来自这个模块。
import torch

# 从 PyTorch 中导入神经网络模块。
# 后面定义 RMSNormReference 类时，需要继承 nn.Module。
from torch import nn


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    使用标准 PyTorch 运算实现 RMSNorm。

    该函数会沿输入张量的最后一个维度执行归一化。

    例如：

        x.shape = [batch_size, hidden_size]

    或者：

        x.shape = [batch_size, sequence_length, hidden_size]

    函数都会沿 hidden_size 所在的最后一个维度计算 RMSNorm。

    Args:
        x:
            输入张量。

            形状必须满足：

                [..., hidden_size]

            其中省略号表示前面可以有任意数量的维度。

        weight:
            RMSNorm 的缩放权重。

            形状必须为：

                [hidden_size]

        eps:
            为了避免除以零而加入的小常数。

            默认值为：

                1e-6

    Returns:
        RMSNorm 的输出张量。

        输出张量的形状、数据类型和设备与 x 保持一致。
    """

    # ------------------------------------------------------------
    # 第一步：检查输入张量 x 的数据类型
    # ------------------------------------------------------------

    # RMSNorm 包含平方、求平均值、开平方和除法，
    # 因此输入必须是浮点数，不能直接使用 int32、int64 等整数类型。
    if not x.is_floating_point():
        raise TypeError(
            "x must be a floating-point tensor, "
            f"but received dtype={x.dtype}"
        )

    # ------------------------------------------------------------
    # 第二步：检查权重 weight 的数据类型
    # ------------------------------------------------------------

    # weight 会与归一化后的 x 相乘，
    # 因此 weight 也必须是浮点张量。
    if not weight.is_floating_point():
        raise TypeError(
            "weight must be a floating-point tensor, "
            f"but received dtype={weight.dtype}"
        )

    # ------------------------------------------------------------
    # 第三步：检查输入张量是否至少有一个维度
    # ------------------------------------------------------------

    # 标量张量的 ndim 为 0，没有“最后一个维度”，
    # 无法执行按 hidden_size 归一化。
    if x.ndim == 0:
        raise ValueError(
            "x must have at least one dimension"
        )

    # ------------------------------------------------------------
    # 第四步：检查 weight 是否是一维张量
    # ------------------------------------------------------------

    # RMSNorm 的 weight 应当为：
    #
    #     [hidden_size]
    #
    # 而不是：
    #
    #     [1, hidden_size]
    #
    # 或其他多维形状。
    if weight.ndim != 1:
        raise ValueError(
            "weight must be a 1-dimensional tensor, "
            f"but received shape={tuple(weight.shape)}"
        )

    # ------------------------------------------------------------
    # 第五步：检查 hidden_size 是否匹配
    # ------------------------------------------------------------

    # x.shape[-1] 表示输入最后一个维度的大小，
    # 也就是需要进行 RMSNorm 的 hidden_size。
    hidden_size = x.shape[-1]

    # weight.numel() 表示 weight 中元素的总数量。
    weight_size = weight.numel()

    # 每个 hidden 维度都需要对应一个缩放权重，
    # 所以 hidden_size 必须与 weight 的元素数量相等。
    if hidden_size != weight_size:
        raise ValueError(
            "The last dimension of x must match weight.numel(): "
            f"x.shape[-1]={hidden_size}, "
            f"weight.numel()={weight_size}"
        )

    # ------------------------------------------------------------
    # 第六步：检查 x 和 weight 是否位于同一个设备
    # ------------------------------------------------------------

    # 例如，不能让 x 位于 GPU，而 weight 位于 CPU。
    #
    # 正确情况：
    #
    #     x.device      == cuda:0
    #     weight.device == cuda:0
    #
    # 错误情况：
    #
    #     x.device      == cuda:0
    #     weight.device == cpu
    if x.device != weight.device:
        raise ValueError(
            "x and weight must be on the same device: "
            f"x.device={x.device}, "
            f"weight.device={weight.device}"
        )

    # ------------------------------------------------------------
    # 第七步：检查 eps 是否有效
    # ------------------------------------------------------------

    # eps 必须大于 0。
    #
    # 如果 eps 小于或等于 0，
    # 当输入全部为 0 时可能出现除以零或非法开方。
    if eps <= 0:
        raise ValueError(
            f"eps must be greater than zero, but received eps={eps}"
        )

    # ------------------------------------------------------------
    # 第八步：选择内部计算使用的数据类型
    # ------------------------------------------------------------

    # FP16 和 BF16 的数值表示范围、精度都比 FP32 低。
    #
    # RMSNorm 中包含：
    #
    #     x²
    #     求和
    #     求平均值
    #     开平方
    #
    # 如果这些步骤全部使用 FP16，容易产生较大的数值误差，
    # 甚至可能发生上溢或下溢。
    #
    # 因此：
    #
    #     FP16 输入  -> 内部使用 FP32 计算
    #     BF16 输入  -> 内部使用 FP32 计算
    #     FP32 输入  -> 继续使用 FP32
    #     FP64 输入  -> 继续使用 FP64
    if x.dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = x.dtype

    # ------------------------------------------------------------
    # 第九步：把 x 转换到内部计算类型
    # ------------------------------------------------------------

    # 如果 x 原本是 FP16，这里会暂时转换为 FP32。
    #
    # 如果 x 原本就是 FP32，则通常不会发生真正的数据复制。
    x_compute = x.to(dtype=compute_dtype)

    # ------------------------------------------------------------
    # 第十步：把 weight 转换到相同计算类型
    # ------------------------------------------------------------

    # 为了让 x_compute 和 weight_compute 能安全相乘，
    # 二者应使用相同的数据类型。
    weight_compute = weight.to(dtype=compute_dtype)

    # ------------------------------------------------------------
    # 第十一步：计算每个元素的平方
    # ------------------------------------------------------------

    # 例如输入一行：
    #
    #     x = [1, 2, 3, 4]
    #
    # 平方后：
    #
    #     x² = [1, 4, 9, 16]
    squared = x_compute.pow(2)

    # ------------------------------------------------------------
    # 第十二步：沿最后一个维度计算平方平均值
    # ------------------------------------------------------------

    # dim=-1 表示沿最后一个维度求平均值。
    #
    # keepdim=True 表示保留最后一个维度，
    # 只是把它的大小变为 1。
    #
    # 例如：
    #
    #     x.shape           = [2, 4096]
    #     mean_square.shape = [2, 1]
    #
    # 保留这个维度后，后面可以利用广播机制，
    # 将每一行的 inverse_rms 乘回整行数据。
    mean_square = squared.mean(
        dim=-1,
        keepdim=True,
    )

    # ------------------------------------------------------------
    # 第十三步：计算均方根倒数
    # ------------------------------------------------------------

    # torch.rsqrt(value) 等价于：
    #
    #     1 / sqrt(value)
    #
    # 因此这里计算的是：
    #
    #     inverse_rms = 1 / sqrt(mean(x²) + eps)
    #
    # 加上 eps 是为了避免输入全为 0 时除以 0。
    inverse_rms = torch.rsqrt(
        mean_square + eps
    )

    # ------------------------------------------------------------
    # 第十四步：对输入执行归一化
    # ------------------------------------------------------------

    # inverse_rms 的最后一个维度为 1，
    # PyTorch 会利用广播机制把它扩展到 hidden_size。
    #
    # 例如：
    #
    #     x_compute.shape = [2, 4096]
    #     inverse_rms.shape = [2, 1]
    #
    # 两者相乘后：
    #
    #     normalized.shape = [2, 4096]
    normalized = x_compute * inverse_rms

    # ------------------------------------------------------------
    # 第十五步：乘以可学习缩放权重
    # ------------------------------------------------------------

    # weight_compute.shape 为：
    #
    #     [hidden_size]
    #
    # PyTorch 同样会使用广播机制，
    # 将 weight 应用到每一行输入上。
    output_compute = normalized * weight_compute

    # ------------------------------------------------------------
    # 第十六步：恢复为输入 x 原本的数据类型
    # ------------------------------------------------------------

    # 如果输入 x 是 FP16：
    #
    #     内部使用 FP32 完成计算
    #     最终再转换回 FP16
    #
    # 这样既能减少计算误差，又能保证输出类型与输入一致。
    output = output_compute.to(dtype=x.dtype)

    # ------------------------------------------------------------
    # 第十七步：返回最终结果
    # ------------------------------------------------------------

    return output


class RMSNormReference(nn.Module):
    """
    使用标准 PyTorch 运算实现的 RMSNorm 模块。

    这个类把 rmsnorm_reference() 函数包装成 nn.Module，
    使用方式与普通 PyTorch 神经网络层类似。

    示例：

        layer = RMSNormReference(hidden_size=4096).cuda()

        x = torch.randn(
            1,
            4096,
            device="cuda",
        )

        output = layer(x)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """
        初始化 RMSNorm 模块。

        Args:
            hidden_size:
                输入最后一个维度的大小。

            eps:
                为避免除以零而加入的小常数。
        """

        # 调用父类 nn.Module 的初始化方法。
        #
        # 所有自定义 PyTorch 模块都应该执行这一行，
        # 否则参数注册、设备迁移等功能可能无法正常工作。
        super().__init__()

        # hidden_size 必须是整数。
        if not isinstance(hidden_size, int):
            raise TypeError(
                "hidden_size must be an integer, "
                f"but received type={type(hidden_size).__name__}"
            )

        # hidden_size 必须大于 0。
        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be greater than zero, "
                f"but received hidden_size={hidden_size}"
            )

        # eps 必须大于 0。
        if eps <= 0:
            raise ValueError(
                f"eps must be greater than zero, but received eps={eps}"
            )

        # 保存 hidden_size，方便外部查看模块配置。
        self.hidden_size = hidden_size

        # 保存数值稳定常数 eps。
        self.eps = eps

        # 创建可学习缩放参数 weight。
        #
        # 初始值全部为 1，因此刚开始时，
        # RMSNorm 只执行归一化，不额外放大或缩小某个维度。
        #
        # 使用 nn.Parameter 包装后，
        # PyTorch 会自动把它注册为模型参数。
        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        定义模块的前向计算过程。

        当执行：

            output = layer(x)

        PyTorch 实际会调用：

            layer.forward(x)
        """

        # 调用前面编写的函数式 RMSNorm 参考实现。
        return rmsnorm_reference(
            x=x,
            weight=self.weight,
            eps=self.eps,
        )