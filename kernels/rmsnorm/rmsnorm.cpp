/*
 * RMSNorm CUDA 扩展的 PyTorch C++ 接口层。
 *
 * 本文件不直接实现 CUDA Kernel。
 *
 * 它主要负责：
 *
 * 1. 接收 Python 传入的 torch.Tensor；
 * 2. 对输入张量、权重和 eps 进行统一检查；
 * 3. 调用不同版本的 CUDA Launcher；
 * 4. 通过 PyBind11 将接口注册给 Python。
 *
 *
 * 当前支持两个 CUDA 实现：
 *
 *     forward(...)
 *         FP32 Naive Shared Memory Reduce 版本。
 *
 *     forward_warp(...)
 *         FP32 Warp Shuffle Reduce 版本。
 *
 *
 * 完整调用关系：
 *
 * Python
 *   ↓
 * rmsnorm.cpp
 *   ├── rmsnorm_cuda_launcher(...)
 *   │       ↓
 *   │   rmsnorm_kernel.cu
 *   │
 *   └── rmsnorm_warp_cuda_launcher(...)
 *           ↓
 *       rmsnorm_warp_kernel.cu
 *           ↓
 *       Jetson Orin GPU
 */


#include <torch/extension.h>


/*
 * 声明 Naive CUDA Launcher。
 *
 * 该函数真正的定义位于：
 *
 *     kernels/rmsnorm/rmsnorm_kernel.cu
 *
 * 当前 C++ 文件只需要知道它的：
 *
 *     - 函数名称；
 *     - 参数类型；
 *     - 返回值类型。
 */
torch::Tensor rmsnorm_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
);


/*
 * 声明 Warp Shuffle CUDA Launcher。
 *
 * 该函数真正的定义位于：
 *
 *     kernels/rmsnorm/rmsnorm_warp_kernel.cu
 */
torch::Tensor rmsnorm_warp_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
);


/*
 * RMSNorm 公共输入检查函数。
 *
 * Naive 版本与 Warp 版本具有相同的输入要求，
 * 因此把检查逻辑提取到一个公共函数中。
 *
 * 这样可以避免：
 *
 *     rmsnorm_forward()
 *
 * 和：
 *
 *     rmsnorm_warp_forward()
 *
 * 各自维护一套重复的检查代码。
 *
 *
 * 参数使用 const 引用：
 *
 *     const torch::Tensor&
 *
 * 表示：
 *
 *     - 不复制 Tensor 对象；
 *     - 当前函数不会修改 Tensor；
 *     - 只读取 Tensor 的属性。
 */
void validate_rmsnorm_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps
) {
    /*
     * 检查一：x 必须位于 CUDA 设备。
     *
     * 自定义 CUDA Kernel 无法直接读取 CPU 内存。
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
     * 检查三：x 和 weight 必须位于同一个设备。
     *
     * 当前 Jetson 一般只有 cuda:0，
     * 但保留该检查能够提高接口完整性。
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
     * RMSNorm 沿最后一个维度执行。
     *
     * 标量张量：
     *
     *     shape = []
     *
     * 不存在最后一个维度，因此不能计算 RMSNorm。
     */
    TORCH_CHECK(
        x.dim() >= 1,
        "x must have at least one dimension"
    );


    /*
     * 检查五：weight 必须是一维张量。
     *
     * 正确形式：
     *
     *     [hidden_size]
     *
     * 错误形式：
     *
     *     [1, hidden_size]
     */
    TORCH_CHECK(
        weight.dim() == 1,
        "weight must be a 1-dimensional tensor, but received dim=",
        weight.dim()
    );


    /*
     * 读取 x 最后一个维度的大小。
     *
     * 例如：
     *
     *     x.shape = [2, 8, 4096]
     *
     * 则：
     *
     *     hidden_size = 4096
     */
    const int64_t hidden_size =
        x.size(-1);


    /*
     * 检查六：hidden_size 必须大于零。
     *
     * 形如：
     *
     *     [2, 0]
     *
     * 的张量不能执行 RMSNorm。
     *
     * 但形如：
     *
     *     [0, 128]
     *
     * 的空行输入是允许的，因为 hidden_size 仍然为 128。
     */
    TORCH_CHECK(
        hidden_size > 0,
        "the last dimension of x must be greater than zero"
    );


    /*
     * 检查七：weight 元素数量必须等于 hidden_size。
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
     * 检查八：当前 CUDA Kernel 只支持 FP32 输入。
     *
     * FP16、BF16、half2 等类型将在后续阶段实现。
     */
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        "the RMSNorm CUDA kernels currently support only float32 x, ",
        "but received dtype=",
        x.scalar_type()
    );


    /*
     * 检查九：weight 当前也只支持 FP32。
     */
    TORCH_CHECK(
        weight.scalar_type() == torch::kFloat32,
        "the RMSNorm CUDA kernels currently support only float32 weight, ",
        "but received dtype=",
        weight.scalar_type()
    );


    /*
     * 检查十：x 必须采用连续内存布局。
     *
     * CUDA Kernel 会按照：
     *
     *     row_offset + column
     *
     * 的方式线性访问数据。
     *
     * 非连续张量的逻辑索引与物理地址不一定连续。
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
     * 检查十二：eps 必须严格大于零。
     *
     * RMSNorm 中会计算：
     *
     *     1 / sqrt(mean_square + eps)
     *
     * 当输入全为零时，mean_square 等于零。
     * eps 必须为正数，才能避免除以零。
     */
    TORCH_CHECK(
        eps > 0.0,
        "eps must be greater than zero, but received eps=",
        eps
    );
}


/*
 * Naive RMSNorm 前向接口。
 *
 * Python 侧对应：
 *
 *     extension.forward(x, weight, eps)
 */
torch::Tensor rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    /*
     * 先执行统一输入检查。
     */
    validate_rmsnorm_inputs(
        x,
        weight,
        eps
    );


    /*
     * 检查通过后，调用 Naive CUDA Launcher。
     */
    return rmsnorm_cuda_launcher(
        x,
        weight,
        eps
    );
}


/*
 * Warp Shuffle RMSNorm 前向接口。
 *
 * Python 侧对应：
 *
 *     extension.forward_warp(x, weight, eps)
 */
torch::Tensor rmsnorm_warp_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    /*
     * Warp 版本与 Naive 版本使用完全相同的输入约束。
     */
    validate_rmsnorm_inputs(
        x,
        weight,
        eps
    );


    /*
     * 检查通过后，调用 Warp Shuffle CUDA Launcher。
     */
    return rmsnorm_warp_cuda_launcher(
        x,
        weight,
        eps
    );
}


/*
 * 使用 PyBind11 注册 Python 扩展接口。
 *
 * TORCH_EXTENSION_NAME 由：
 *
 *     torch.utils.cpp_extension.load()
 *
 * 在编译阶段自动提供。
 */
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    /*
     * 注册 Naive CUDA 实现。
     *
     * Python 调用形式：
     *
     *     extension.forward(
     *         x,
     *         weight,
     *         eps,
     *     )
     */
    module.def(
        "forward",
        &rmsnorm_forward,
        "RMSNorm naive CUDA forward"
    );


    /*
     * 注册 Warp Shuffle CUDA 实现。
     *
     * Python 调用形式：
     *
     *     extension.forward_warp(
     *         x,
     *         weight,
     *         eps,
     *     )
     */
    module.def(
        "forward_warp",
        &rmsnorm_warp_forward,
        "RMSNorm warp shuffle CUDA forward"
    );
}
