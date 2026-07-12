/*
 * RMSNorm Naive CUDA Kernel。
 *
 * 这个文件负责真正的 GPU 计算。
 *
 * 当前版本的设计目标：
 *
 * 1. 先保证计算逻辑正确；
 * 2. 支持任意前置维度的输入；
 * 3. 沿输入最后一个维度执行 RMSNorm；
 * 4. 暂时只支持 float32；
 * 5. 使用一个 CUDA Block 处理一行数据；
 * 6. 使用 Shared Memory 完成平方和规约。
 *
 * 例如输入形状：
 *
 *     [batch_size, hidden_size]
 *
 * 或：
 *
 *     [batch_size, sequence_length, hidden_size]
 *
 * 在 Kernel 内都会被视为：
 *
 *     [rows, hidden_size]
 *
 * 其中：
 *
 *     rows = x.numel() / hidden_size
 *
 * RMSNorm 公式：
 *
 *     mean_square = sum(x_i²) / hidden_size
 *
 *     inverse_rms = 1 / sqrt(mean_square + eps)
 *
 *     output_i = x_i * inverse_rms * weight_i
 */

#include <torch/extension.h>

/*
 * PyTorch CUDA 上下文相关接口。
 *
 * 用于取得当前 PyTorch CUDA Stream。
 */
#include <ATen/cuda/CUDAContext.h>

/*
 * CUDAGuard 用于保证 Kernel 在输入张量所属的 CUDA 设备上运行。
 */
#include <c10/cuda/CUDAGuard.h>

/*
 * C10_CUDA_KERNEL_LAUNCH_CHECK() 定义在这里。
 *
 * 它可以在 Kernel 启动后检查是否出现：
 *
 *     invalid configuration argument
 *     illegal memory access
 *     invalid device function
 *
 * 等 CUDA 错误。
 */
#include <c10/cuda/CUDAException.h>

/*
 * CUDA Runtime API。
 *
 * 包含：
 *
 *     cudaStream_t
 *     __global__
 *     threadIdx
 *     blockIdx
 *     __syncthreads()
 *     rsqrtf()
 */
#include <cuda.h>
#include <cuda_runtime.h>


/*
 * RMSNorm Naive CUDA Kernel。
 *
 * Kernel 的线程组织方式：
 *
 *     一个 CUDA Block 负责一行输入；
 *     一个 Block 内有多个线程；
 *     每个线程处理该行中的若干元素。
 *
 * 参数：
 *
 *     x:
 *         输入张量的设备内存指针。
 *
 *     weight:
 *         RMSNorm 权重的设备内存指针。
 *
 *     output:
 *         输出张量的设备内存指针。
 *
 *     rows:
 *         输入被展平后的行数。
 *
 *     hidden_size:
 *         每一行包含的元素数量。
 *
 *     eps:
 *         防止除以零的数值稳定常数。
 */
__global__ void rmsnorm_naive_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t rows,
    int64_t hidden_size,
    float eps
) {
    /*
     * 动态 Shared Memory。
     *
     * 每个线程会先计算一部分平方和，
     * 然后把自己的局部结果写入 shared_sum。
     *
     * 实际空间大小在启动 Kernel 时指定：
     *
     *     threads_per_block * sizeof(float)
     */
    extern __shared__ float shared_sum[];

    /*
     * 当前线程在线程块中的编号。
     *
     * 如果每个 Block 有 256 个线程：
     *
     *     thread_id 的范围为 0 到 255。
     */
    const int thread_id = threadIdx.x;

    /*
     * 当前 Block 对应的行号。
     *
     * 一个 Block 负责一行，所以：
     *
     *     blockIdx.x = 0  -> 第 0 行
     *     blockIdx.x = 1  -> 第 1 行
     *     blockIdx.x = 2  -> 第 2 行
     */
    const int64_t row_id =
        static_cast<int64_t>(blockIdx.x);

    /*
     * 理论上 gridDim.x 会设置为 rows，
     * 因此 row_id 通常不会越界。
     *
     * 保留这个判断可以提高 Kernel 的安全性。
     */
    if (row_id >= rows) {
        return;
    }

    /*
     * 计算当前行在一维连续内存中的起始位置。
     *
     * 例如：
     *
     *     hidden_size = 4096
     *     row_id = 2
     *
     * 则当前行的起始元素索引为：
     *
     *     2 * 4096 = 8192
     */
    const int64_t row_offset =
        row_id * hidden_size;

    /*
     * 得到当前输入行的指针。
     *
     * 后面可以直接使用：
     *
     *     row_input[column]
     *
     * 访问这一行中的元素。
     */
    const float* row_input =
        x + row_offset;

    /*
     * 得到当前输出行的指针。
     */
    float* row_output =
        output + row_offset;

    /*
     * 保存当前线程计算得到的局部平方和。
     *
     * 每个线程只处理当前行中的一部分元素。
     */
    float local_square_sum = 0.0f;

    /*
     * 每个线程以 blockDim.x 为步长遍历当前行。
     *
     * 假设：
     *
     *     hidden_size = 1024
     *     blockDim.x = 256
     *
     * 那么：
     *
     *     线程 0 处理索引 0、256、512、768
     *     线程 1 处理索引 1、257、513、769
     *     线程 2 处理索引 2、258、514、770
     *
     * 这种访问方式能够让相邻线程读取相邻内存，
     * 有利于 Global Memory 合并访问。
     */
    for (
        int64_t column = thread_id;
        column < hidden_size;
        column += blockDim.x
    ) {
        /*
         * 从 Global Memory 读取输入元素。
         */
        const float value =
            row_input[column];

        /*
         * 将当前元素的平方累加到线程局部变量中。
         */
        local_square_sum +=
            value * value;
    }

    /*
     * 每个线程把自己的局部平方和写入 Shared Memory。
     *
     * thread_id 不同，因此每个线程写入不同位置。
     */
    shared_sum[thread_id] =
        local_square_sum;

    /*
     * 等待同一个 Block 内所有线程完成写入。
     *
     * 如果没有这个同步，
     * 某些线程可能在其他线程尚未写入时就开始规约。
     */
    __syncthreads();

    /*
     * 使用二叉树方式进行 Shared Memory 规约。
     *
     * 假设 blockDim.x = 256：
     *
     * 第一轮：
     *
     *     0 号线程加上 128 号线程
     *     1 号线程加上 129 号线程
     *     ...
     *
     * 第二轮：
     *
     *     0 号线程加上 64 号线程
     *     1 号线程加上 65 号线程
     *
     * 依次缩小：
     *
     *     128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
     *
     * 最终：
     *
     *     shared_sum[0]
     *
     * 保存整行所有元素的平方和。
     */
    for (
        unsigned int stride =
            blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        /*
         * 只有前 stride 个线程参与当前轮规约。
         */
        if (
            static_cast<unsigned int>(thread_id)
            < stride
        ) {
            shared_sum[thread_id] +=
                shared_sum[
                    thread_id + stride
                ];
        }

        /*
         * 每一轮规约结束后必须同步。
         *
         * 保证下一轮读取到的是当前轮已经更新后的结果。
         */
        __syncthreads();
    }

    /*
     * 规约完成后：
     *
     *     shared_sum[0]
     *
     * 等于当前整行的平方和。
     *
     * 所有线程都会读取同一个 inverse_rms，
     * 因此可以由每个线程直接计算。
     *
     * mean_square：
     *
     *     sum(x²) / hidden_size
     */
    const float mean_square =
        shared_sum[0]
        / static_cast<float>(hidden_size);

    /*
     * rsqrtf(value) 等价于：
     *
     *     1.0f / sqrtf(value)
     *
     * 使用 rsqrtf 可以直接得到 RMS 的倒数。
     */
    const float inverse_rms =
        rsqrtf(
            mean_square + eps
        );

    /*
     * 第二次遍历当前行。
     *
     * 每个线程再次处理自己负责的列：
     *
     *     output_i = x_i * inverse_rms * weight_i
     */
    for (
        int64_t column = thread_id;
        column < hidden_size;
        column += blockDim.x
    ) {
        /*
         * weight 只有一个 hidden_size 维度。
         *
         * 所有输入行共享同一组 weight。
         */
        row_output[column] =
            row_input[column]
            * inverse_rms
            * weight[column];
    }
}


/*
 * CUDA Kernel 启动函数。
 *
 * 这个函数会被 rmsnorm.cpp 中的：
 *
 *     rmsnorm_forward()
 *
 * 调用。
 *
 * 它负责：
 *
 * 1. 创建输出张量；
 * 2. 计算 rows 和 hidden_size；
 * 3. 设置 Grid、Block 和 Shared Memory；
 * 4. 获取当前 PyTorch CUDA Stream；
 * 5. 启动 rmsnorm_naive_kernel；
 * 6. 检查 Kernel 启动错误；
 * 7. 返回输出张量。
 */
torch::Tensor rmsnorm_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    /*
     * CUDAGuard 确保当前 CUDA 设备与 x 所在设备一致。
     *
     * 在当前 Jetson 上一般只有 cuda:0，
     * 但保留它可以让代码更规范。
     */
    const c10::cuda::CUDAGuard device_guard(
        x.device()
    );

    /*
     * 创建与 x 完全相同属性的输出张量。
     *
     * empty_like 会继承：
     *
     *     shape
     *     dtype
     *     device
     *     memory format
     */
    torch::Tensor output =
        torch::empty_like(x);

    /*
     * 读取输入最后一个维度。
     *
     * 这就是 RMSNorm 的 hidden_size。
     */
    const int64_t hidden_size =
        x.size(-1);

    /*
     * 将所有前置维度展平为 rows。
     *
     * 例如：
     *
     *     x.shape = [2, 3, 4096]
     *
     * 那么：
     *
     *     x.numel() = 2 * 3 * 4096
     *
     *     rows = x.numel() / 4096
     *          = 6
     */
    const int64_t rows =
        x.numel() / hidden_size;

    /*
     * 如果输入包含零行：
     *
     *     x.shape = [0, hidden_size]
     *
     * 则不需要启动 CUDA Kernel，
     * 直接返回空输出即可。
     */
    if (rows == 0) {
        return output;
    }

    /*
     * 每个 CUDA Block 使用 256 个线程。
     *
     * 256 是常见的起始配置：
     *
     *     256 / 32 = 8 个 Warp
     *
     * 当前 Naive 版本固定使用 256，
     * 后续 Benchmark 阶段再比较 64、128、256、512。
     */
    constexpr int threads_per_block =
        256;

    /*
     * 一个 Block 负责一行，因此 Block 数量等于 rows。
     */
    const int64_t blocks =
        rows;

    /*
     * 每个线程需要在 Shared Memory 中保存一个 float。
     *
     * Shared Memory 大小为：
     *
     *     256 * sizeof(float)
     *
     * 即：
     *
     *     256 * 4 = 1024 bytes
     */
    const std::size_t shared_memory_bytes =
        threads_per_block * sizeof(float);

    /*
     * 取得当前 PyTorch CUDA Stream。
     *
     * 自定义 Kernel 必须运行在 PyTorch 当前 Stream 上，
     * 才能与其他 PyTorch CUDA 操作保持正确的执行顺序。
     */
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    /*
     * 从 torch.Tensor 中取得底层设备内存指针。
     *
     * 当前版本已经在 rmsnorm.cpp 中检查：
     *
     *     x.dtype == float32
     *     weight.dtype == float32
     *
     * 因此可以安全使用 data_ptr<float>()。
     */
    const float* x_data =
        x.data_ptr<float>();

    const float* weight_data =
        weight.data_ptr<float>();

    float* output_data =
        output.data_ptr<float>();

    /*
     * 启动 CUDA Kernel。
     *
     * <<<blocks, threads, shared_memory, stream>>>
     *
     * 四个启动参数依次表示：
     *
     *     Grid 中 Block 数量；
     *     每个 Block 的线程数量；
     *     每个 Block 的动态 Shared Memory 大小；
     *     Kernel 使用的 CUDA Stream。
     */
    rmsnorm_naive_kernel<<<
        static_cast<unsigned int>(blocks),
        threads_per_block,
        shared_memory_bytes,
        stream
    >>>(
        x_data,
        weight_data,
        output_data,
        rows,
        hidden_size,
        static_cast<float>(eps)
    );

    /*
     * 检查 Kernel 启动是否发生错误。
     *
     * 这里只检查启动错误，
     * 不会在每次调用后强制 cudaDeviceSynchronize()。
     *
     * 避免不必要的全局同步，
     * 保持 PyTorch CUDA 的异步执行特性。
     */
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    /*
     * 返回 CUDA 输出张量。
     */
    return output;
}
