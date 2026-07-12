/*
 * RMSNorm CUDA 扩展的 PyTorch C++ 接口层。
 *
 * 这个文件本身不包含 CUDA Kernel。
 *
 * 它主要负责：
 *
 * 1. 接收 Python 传入的 torch.Tensor；
 * 2. 检查输入张量是否合法；
 * 3. 调用 rmsnorm_kernel.cu 中真正的 CUDA 实现；
 * 4. 使用 pybind11 把 C++ 函数暴露给 Python。
 *
 * 整体调用关系：
 *
 * Python
 *   ↓
 * rmsnorm.cpp
 *   ↓
 * rmsnorm_kernel.cu
 *   ↓
 * Jetson Orin GPU
 */

#include <torch/extension.h>


/*
 * 声明 CUDA 启动函数。
 *
 * 这个函数会在 rmsnorm_kernel.cu 中实现。
 *
 * 这里只写函数声明，是为了让当前 C++ 文件知道：
 *
 *     rmsnorm_cuda_launcher(...)
 *
 * 这个函数存在，并且可以被调用。
 *
 * 参数：
 *
 *     x:
 *         输入 CUDA 张量。
 *
 *     weight:
 *         RMSNorm 的缩放权重。
 *
 *     eps:
 *         防止除以零的数值稳定常数。
 *
 * 返回值：
 *
 *     与 x 形状相同的 CUDA 输出张量。
 */
torch::Tensor rmsnorm_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
);


/*
 * RMSNorm CUDA 前向接口。
 *
 * Python 最终会调用这个函数。
 *
 * 该函数不会直接执行 GPU 计算，而是先完成输入检查，
 * 确认参数正确后，再调用 rmsnorm_cuda_launcher()。
 */
torch::Tensor rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    /*
     * 检查一：x 必须位于 CUDA 设备。
     *
     * 自定义 CUDA Kernel 不能直接处理 CPU 张量。
     *
     * 正确示例：
     *
     *     x.device == cuda:0
     *
     * 错误示例：
     *
     *     x.device == cpu
     */
    TORCH_CHECK(
        x.is_cuda(),
        "x must be a CUDA tensor, but received device=",
        x.device()
    );

    /*
     * 检查二：weight 也必须位于 CUDA 设备。
     */
    TORCH_CHECK(
        weight.is_cuda(),
        "weight must be a CUDA tensor, but received device=",
        weight.device()
    );

    /*
     * 检查三：x 和 weight 必须位于同一个 CUDA 设备。
     *
     * 当前 Jetson 只有一个 GPU，但仍然保留这个检查，
     * 这样接口逻辑更完整，也便于以后迁移到多 GPU 环境。
     */
    TORCH_CHECK(
        x.device() == weight.device(),
        "x and weight must be on the same device: ",
        "x.device=",
        x.device(),
        ", weight.device=",
        weight.device()
    );

    /*
     * 检查四：x 至少应有一个维度。
     *
     * RMSNorm 会沿最后一个维度进行归一化。
     * 标量张量没有最后一个维度，因此不能参与计算。
     */
    TORCH_CHECK(
        x.dim() >= 1,
        "x must have at least one dimension"
    );

    /*
     * 检查五：weight 必须是一维张量。
     *
     * 正确形状：
     *
     *     [hidden_size]
     *
     * 错误形状：
     *
     *     [1, hidden_size]
     */
    TORCH_CHECK(
        weight.dim() == 1,
        "weight must be a 1-dimensional tensor, but received dim=",
        weight.dim()
    );

    /*
     * 读取输入最后一个维度的大小。
     *
     * 例如：
     *
     *     x.shape = [2, 3, 4096]
     *
     * 则：
     *
     *     hidden_size = 4096
     */
    const int64_t hidden_size = x.size(-1);

    /*
     * 检查六：hidden_size 必须大于零。
     */
    TORCH_CHECK(
        hidden_size > 0,
        "the last dimension of x must be greater than zero"
    );

    /*
     * 检查七：weight 的元素数量必须等于 hidden_size。
     *
     * 每个隐藏维度都需要一个对应的缩放权重。
     */
    TORCH_CHECK(
        weight.numel() == hidden_size,
        "the last dimension of x must match weight.numel(): ",
        "x.size(-1)=",
        hidden_size,
        ", weight.numel()=",
        weight.numel()
    );

    /*
     * 检查八：Naive 版本暂时只支持 FP32 输入。
     *
     * 我们先把最简单、最容易验证的 FP32 Kernel 跑通。
     *
     * FP16、half2 和向量化优化会在后续阶段实现。
     */
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        "the naive RMSNorm CUDA kernel currently supports only float32 x, ",
        "but received dtype=",
        x.scalar_type()
    );

    /*
     * 检查九：weight 同样暂时只支持 FP32。
     */
    TORCH_CHECK(
        weight.scalar_type() == torch::kFloat32,
        "the naive RMSNorm CUDA kernel currently supports only float32 weight, ",
        "but received dtype=",
        weight.scalar_type()
    );

    /*
     * 检查十：x 必须在内存中连续。
     *
     * CUDA Kernel 会按照线性地址读取数据。
     *
     * 如果张量不是 contiguous，
     * 它的逻辑索引和物理内存位置可能不连续。
     */
    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous"
    );

    /*
     * 检查十一：weight 也必须连续。
     */
    TORCH_CHECK(
        weight.is_contiguous(),
        "weight must be contiguous"
    );

    /*
     * 检查十二：eps 必须大于零。
     *
     * 当输入全为零时：
     *
     *     mean(x²) = 0
     *
     * 如果 eps 也为零，就会出现除以零。
     */
    TORCH_CHECK(
        eps > 0.0,
        "eps must be greater than zero, but received eps=",
        eps
    );

    /*
     * 所有输入检查通过后，
     * 调用 rmsnorm_kernel.cu 中的 CUDA 启动函数。
     */
    return rmsnorm_cuda_launcher(
        x,
        weight,
        eps
    );
}


/*
 * PYBIND11_MODULE 用于把 C++ 函数注册成 Python 模块接口。
 *
 * TORCH_EXTENSION_NAME 由 PyTorch 编译扩展时自动提供。
 *
 * 注册完成后，Python 侧可以使用类似方式调用：
 *
 *     extension.forward(x, weight, eps)
 */
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    /*
     * 第一个参数 "forward"：
     *
     *     Python 侧看到的函数名称。
     *
     * 第二个参数 &rmsnorm_forward：
     *
     *     对应的 C++ 函数地址。
     *
     * 第三个参数：
     *
     *     接口说明文字。
     */
    module.def(
        "forward",
        &rmsnorm_forward,
        "RMSNorm naive CUDA forward"
    );
}
