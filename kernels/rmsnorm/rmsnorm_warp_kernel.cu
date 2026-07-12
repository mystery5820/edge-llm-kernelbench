/*
 * RMSNorm Warp Shuffle Reduce CUDA Kernel。
 *
 * 本文件实现 RMSNorm 的第二个 CUDA 版本：
 *
 *     Warp Shuffle Reduce 版本
 *
 * 它是在 Naive Shared Memory Reduce 版本基础上的第一次优化。
 *
 *
 * 一、RMSNorm 计算公式
 * -------------------------------------------------------------------------
 *
 * 对输入向量：
 *
 *     x = [x_0, x_1, ..., x_(hidden_size - 1)]
 *
 * 首先计算平方均值：
 *
 *     mean_square =
 *         (x_0² + x_1² + ... + x_(hidden_size - 1)²)
 *         / hidden_size
 *
 * 然后计算 RMS 的倒数：
 *
 *     inverse_rms =
 *         1 / sqrt(mean_square + eps)
 *
 * 最后进行归一化和权重缩放：
 *
 *     output_i =
 *         x_i * inverse_rms * weight_i
 *
 *
 * 二、输入张量的逻辑形状
 * -------------------------------------------------------------------------
 *
 * Python 侧输入可能是：
 *
 *     [hidden_size]
 *
 *     [batch_size, hidden_size]
 *
 *     [batch_size, sequence_length, hidden_size]
 *
 * 进入 CUDA Kernel 后，所有前置维度都被展平：
 *
 *     [rows, hidden_size]
 *
 * 其中：
 *
 *     rows =
 *         x.numel() / hidden_size
 *
 *
 * 三、线程组织方式
 * -------------------------------------------------------------------------
 *
 * 当前版本仍然采用：
 *
 *     一个 CUDA Block 处理一行输入。
 *
 * 每个 Block 固定使用：
 *
 *     256 个线程
 *
 * NVIDIA GPU 中：
 *
 *     一个 Warp = 32 个线程
 *
 * 所以：
 *
 *     256 个线程 = 8 个 Warp
 *
 *
 * 四、为什么使用 Warp Shuffle
 * -------------------------------------------------------------------------
 *
 * Naive 版本中，每一个线程都需要：
 *
 *     1. 将局部平方和写入 Shared Memory；
 *     2. 多轮读取和写入 Shared Memory；
 *     3. 每轮执行 __syncthreads()。
 *
 * 如果一个 Block 有 256 个线程，规约阶段需要经历：
 *
 *     128
 *     64
 *     32
 *     16
 *     8
 *     4
 *     2
 *     1
 *
 * 多轮 Shared Memory 访问和 Block 级同步会带来额外开销。
 *
 * Warp Shuffle 指令允许同一个 Warp 中的线程直接交换寄存器数据：
 *
 *     __shfl_down_sync(...)
 *
 * 不需要：
 *
 *     - 将数据写入 Shared Memory；
 *     - 执行 __syncthreads()；
 *     - 再从 Shared Memory 读取。
 *
 *
 * 五、本版本的两级规约过程
 * -------------------------------------------------------------------------
 *
 * 第一级：Warp 内规约
 *
 *     每个 Warp 内部的 32 个线程，
 *     使用 __shfl_down_sync() 计算自己的 Warp 平方和。
 *
 *     256 个线程一共有 8 个 Warp，
 *     所以第一级结束后得到 8 个 Warp Sum。
 *
 * 第二级：Warp 间规约
 *
 *     每个 Warp 的 lane 0，
 *     将 Warp Sum 写入 Shared Memory。
 *
 *     Shared Memory 中只有 8 个有效值。
 *
 *     然后由第一个 Warp 读取这 8 个值，
 *     再执行一次 Warp Shuffle Reduce。
 *
 *     最终得到整行的平方和。
 *
 *
 * 六、当前版本范围
 * -------------------------------------------------------------------------
 *
 * 当前 Warp 版本：
 *
 *     - 只支持 float32；
 *     - 一个 Block 处理一行；
 *     - 每个 Block 固定 256 个线程；
 *     - 沿最后一个维度执行 RMSNorm；
 *     - 使用 Warp Shuffle 完成平方和规约；
 *     - 输出形状、dtype 和 device 与输入一致。
 *
 * 后续还可以继续优化：
 *
 *     - float4 向量化读取；
 *     - FP16 输入和 FP32 累加；
 *     - half2 向量化；
 *     - 一个 Warp 处理一行；
 *     - 一个 Block 处理多行；
 *     - 融合加载、规约与输出策略。
 */


/*
 * PyTorch C++ Tensor 接口。
 *
 * 本文件中的 Launcher 返回 torch::Tensor，
 * 因此需要包含 PyTorch Extension 头文件。
 */
#include <torch/extension.h>


/*
 * PyTorch CUDA Context 接口。
 *
 * 用于获取当前 PyTorch CUDA Stream：
 *
 *     at::cuda::getCurrentCUDAStream()
 */
#include <ATen/cuda/CUDAContext.h>


/*
 * CUDA Device Guard。
 *
 * 保证当前 CUDA 设备与输入张量所在设备一致。
 */
#include <c10/cuda/CUDAGuard.h>


/*
 * CUDA Kernel 启动错误检查。
 *
 * 提供：
 *
 *     C10_CUDA_KERNEL_LAUNCH_CHECK()
 */
#include <c10/cuda/CUDAException.h>


/*
 * CUDA Driver 和 CUDA Runtime 头文件。
 *
 * 提供：
 *
 *     __global__
 *     __device__
 *     __shared__
 *     __shfl_down_sync
 *     __syncthreads
 *     blockIdx
 *     threadIdx
 *     blockDim
 *     cudaStream_t
 *     rsqrtf
 */
#include <cuda.h>
#include <cuda_runtime.h>


/*
 * 标准整数类型。
 *
 * 用于：
 *
 *     int64_t
 */
#include <cstdint>


/*
 * 标准 size 类型。
 *
 * 用于：
 *
 *     std::size_t
 */
#include <cstddef>


/*
 * NVIDIA GPU 的 Warp 大小固定为 32。
 *
 * 当前目标设备 Jetson Orin Nano 的 GPU 架构为 Ampere，
 * Warp 同样由 32 个线程组成。
 */
constexpr int kWarpSize = 32;


/*
 * 完整 Warp 的线程掩码。
 *
 * 二进制表示为 32 个 1：
 *
 *     11111111111111111111111111111111
 *
 * 十六进制表示为：
 *
 *     0xffffffff
 *
 * 使用这个掩码表示：
 *
 *     当前 Warp 中 32 个线程都参与 Shuffle 操作。
 */
constexpr unsigned int kFullWarpMask =
    0xffffffffu;


/*
 * Warp 内求和函数。
 *
 * __device__：
 *
 *     这个函数只能在 GPU 设备代码中调用。
 *
 * __forceinline__：
 *
 *     建议编译器把函数直接展开到调用位置，
 *     避免普通函数调用的额外开销。
 *
 *
 * 参数 value：
 *
 *     当前线程持有的局部数值。
 *
 *
 * 返回值：
 *
 *     对 lane 0 来说，返回整个 Warp 的总和。
 *
 *     对其他 lane 来说，最终值不一定是完整总和，
 *     但在本 Kernel 中只会使用 lane 0 的结果。
 */
__device__ __forceinline__ float warp_reduce_sum(
    float value
) {
    /*
     * 一个 Warp 有 32 个线程。
     *
     * 规约过程：
     *
     *     offset = 16
     *     offset = 8
     *     offset = 4
     *     offset = 2
     *     offset = 1
     *
     * 每一轮中，当前线程从：
     *
     *     lane_id + offset
     *
     * 对应的线程读取 value。
     *
     *
     * 以 offset = 16 为例：
     *
     *     lane 0 读取 lane 16
     *     lane 1 读取 lane 17
     *     ...
     *     lane 15 读取 lane 31
     *
     * 然后把读取到的值累加到自己的 value 中。
     */
    for (
        int offset = kWarpSize / 2;
        offset > 0;
        offset /= 2
    ) {
        /*
         * __shfl_down_sync 的参数：
         *
         * 第一个参数：
         *
         *     参与操作的线程掩码。
         *
         * 第二个参数：
         *
         *     当前线程的 value。
         *
         * 第三个参数：
         *
         *     从 lane_id + offset 的线程读取。
         *
         * 返回值：
         *
         *     从目标线程寄存器中取得的数据。
         */
        value += __shfl_down_sync(
            kFullWarpMask,
            value,
            offset
        );
    }

    /*
     * 规约结束后：
     *
     *     lane 0 中保存整个 Warp 的总和。
     */
    return value;
}


/*
 * RMSNorm Warp Reduce CUDA Kernel。
 *
 *
 * 参数 x：
 *
 *     输入张量的 CUDA 设备内存指针。
 *
 * 参数 weight：
 *
 *     RMSNorm 缩放权重。
 *
 *     所有输入行共享同一组 weight。
 *
 * 参数 output：
 *
 *     输出张量的 CUDA 设备内存指针。
 *
 * 参数 rows：
 *
 *     输入张量展平后的行数。
 *
 * 参数 hidden_size：
 *
 *     每一行的元素数量。
 *
 * 参数 eps：
 *
 *     防止除以零的数值稳定常数。
 */
__global__ void rmsnorm_warp_kernel(
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
     * 与 Naive 版本不同，
     * 这里不需要为每一个线程分配一个 float。
     *
     * 一个 Block 有 256 个线程：
     *
     *     256 / 32 = 8 个 Warp
     *
     * 每个 Warp 只写入一个 Warp Sum，
     * 所以 Shared Memory 只需要保存 8 个 float。
     *
     * 实际大小由 Launcher 指定：
     *
     *     warp_count * sizeof(float)
     */
    extern __shared__ float shared_warp_sums[];


    /*
     * 保存整行的 inverse_rms。
     *
     * 由 Block 中的一个线程计算，
     * 然后供整个 Block 的所有线程读取。
     *
     * 这是静态 Shared Memory，
     * 只占用一个 float，也就是 4 字节。
     */
    __shared__ float shared_inverse_rms;


    /*
     * 当前线程在 Block 内的线性编号。
     *
     * 当前 Block 固定有 256 个线程，
     * 因此范围为：
     *
     *     0 到 255
     */
    const int thread_id =
        static_cast<int>(threadIdx.x);


    /*
     * 当前线程在所属 Warp 内的编号。
     *
     * 一个 Warp 有 32 个线程。
     *
     * lane_id 范围为：
     *
     *     0 到 31
     *
     * 由于 32 是 2 的 5 次方：
     *
     *     thread_id % 32
     *
     * 可以写成位运算：
     *
     *     thread_id & 31
     */
    const int lane_id =
        thread_id & (kWarpSize - 1);


    /*
     * 当前线程属于 Block 中的第几个 Warp。
     *
     * 因为一个 Warp 有 32 个线程：
     *
     *     warp_id = thread_id / 32
     *
     * 除以 32 可以写成右移 5 位：
     *
     *     thread_id >> 5
     *
     * 当 Block 有 256 个线程时：
     *
     *     warp_id 范围为 0 到 7。
     */
    const int warp_id =
        thread_id / kWarpSize;


    /*
     * 当前 Block 中包含的 Warp 数量。
     *
     * 通用计算公式：
     *
     *     ceil(blockDim.x / 32)
     *
     * 用整数运算表示为：
     *
     *     (blockDim.x + 31) / 32
     *
     * 当前 blockDim.x 固定为 256，
     * 所以：
     *
     *     warp_count = 8
     */
    const int warp_count =
        (
            static_cast<int>(blockDim.x)
            + kWarpSize
            - 1
        )
        / kWarpSize;


    /*
     * 一个 Block 处理一行，
     * 所以 blockIdx.x 就是当前行号。
     */
    const int64_t row_id =
        static_cast<int64_t>(blockIdx.x);


    /*
     * 防御性越界检查。
     *
     * Launcher 正常情况下会设置：
     *
     *     gridDim.x = rows
     *
     * 因此 row_id 不应该超过 rows。
     *
     * 因为同一个 Block 的所有线程拥有相同的 blockIdx.x，
     * 所以要么整个 Block 都返回，要么整个 Block 都继续执行。
     *
     * 这种条件分支不会导致后续 __syncthreads() 死锁。
     */
    if (row_id >= rows) {
        return;
    }


    /*
     * 计算当前行在连续内存中的起始偏移。
     *
     * 例如：
     *
     *     hidden_size = 4096
     *     row_id = 3
     *
     * 则：
     *
     *     row_offset = 3 * 4096
     *                = 12288
     */
    const int64_t row_offset =
        row_id * hidden_size;


    /*
     * 当前输入行的起始指针。
     */
    const float* row_input =
        x + row_offset;


    /*
     * 当前输出行的起始指针。
     */
    float* row_output =
        output + row_offset;


    /*
     * 每个线程的局部平方和。
     *
     * 初始值为零。
     */
    float local_square_sum =
        0.0f;


    /*
     * 每个线程以 blockDim.x 为步长，
     * 遍历当前行中属于自己的元素。
     *
     *
     * 假设：
     *
     *     hidden_size = 4096
     *     blockDim.x = 256
     *
     * 则线程 0 处理：
     *
     *     0
     *     256
     *     512
     *     ...
     *     3840
     *
     * 线程 1 处理：
     *
     *     1
     *     257
     *     513
     *     ...
     *     3841
     *
     *
     * 相邻线程访问相邻元素，
     * 有利于形成合并的 Global Memory 访问。
     */
    for (
        int64_t column = thread_id;
        column < hidden_size;
        column += blockDim.x
    ) {
        /*
         * 从 Global Memory 读取当前输入元素。
         */
        const float value =
            row_input[column];


        /*
         * 将平方累加到线程局部寄存器中。
         *
         * local_square_sum 通常保存在寄存器里，
         * 不需要在每次循环中访问 Shared Memory。
         */
        local_square_sum +=
            value * value;
    }


    /*
     * 第一级规约：
     *
     * 在每一个 Warp 内部，
     * 使用 Shuffle 指令对 local_square_sum 求和。
     *
     * 调用结束后：
     *
     *     每个 Warp 的 lane 0
     *
     * 保存该 Warp 32 个线程的平方和。
     */
    const float warp_square_sum =
        warp_reduce_sum(
            local_square_sum
        );


    /*
     * 每个 Warp 只有 lane 0 负责写入 Shared Memory。
     *
     * 当前共有 8 个 Warp，
     * 所以只有 8 个线程执行写入：
     *
     *     Warp 0 的 lane 0
     *     Warp 1 的 lane 0
     *     ...
     *     Warp 7 的 lane 0
     *
     * 对应写入：
     *
     *     shared_warp_sums[0]
     *     shared_warp_sums[1]
     *     ...
     *     shared_warp_sums[7]
     */
    if (lane_id == 0) {
        shared_warp_sums[warp_id] =
            warp_square_sum;
    }


    /*
     * Block 级同步。
     *
     * 必须确保所有 Warp 的 lane 0
     * 都已经完成 Shared Memory 写入，
     * 第一个 Warp 才能开始读取这些结果。
     *
     * 与 Naive 版本相比，
     * 这里只在 Warp 间交换结果时需要 Block 级同步。
     */
    __syncthreads();


    /*
     * 第二级规约：
     *
     * 只有第一个 Warp，也就是 warp_id == 0，
     * 负责汇总所有 Warp 的平方和。
     */
    if (warp_id == 0) {
        /*
         * 第一个 Warp 有 32 个线程，
         * 但当前 Block 只有 8 个 Warp Sum。
         *
         * 因此：
         *
         *     lane 0 读取 shared_warp_sums[0]
         *     lane 1 读取 shared_warp_sums[1]
         *     ...
         *     lane 7 读取 shared_warp_sums[7]
         *
         *     lane 8 到 lane 31 使用 0。
         *
         * 这样 32 个线程都可以安全参与完整 Warp Shuffle。
         */
        float block_square_sum =
            lane_id < warp_count
                ? shared_warp_sums[lane_id]
                : 0.0f;


        /*
         * 第一个 Warp 再执行一次 Warp Reduce。
         *
         * 规约结束后：
         *
         *     第一个 Warp 的 lane 0
         *
         * 保存整个 Block，也就是整行输入的平方和。
         */
        block_square_sum =
            warp_reduce_sum(
                block_square_sum
            );


        /*
         * 只有第一个 Warp 的 lane 0，
         * 负责计算整行的 inverse_rms。
         */
        if (lane_id == 0) {
            /*
             * 平方均值：
             *
             *     mean_square =
             *         sum(x_i²) / hidden_size
             */
            const float mean_square =
                block_square_sum
                / static_cast<float>(
                    hidden_size
                );


            /*
             * rsqrtf(value) 等价于：
             *
             *     1.0f / sqrtf(value)
             *
             * 直接得到 RMS 的倒数。
             */
            shared_inverse_rms =
                rsqrtf(
                    mean_square + eps
                );
        }
    }


    /*
     * 第二次 Block 级同步。
     *
     * 确保第一个 Warp 的 lane 0
     * 已经写入 shared_inverse_rms。
     *
     * 然后整个 Block 的所有线程
     * 才能进入输出计算阶段。
     */
    __syncthreads();


    /*
     * 将 Shared Memory 中的 inverse_rms
     * 读取到线程局部变量。
     *
     * 这样后面的循环可以直接使用局部变量。
     */
    const float inverse_rms =
        shared_inverse_rms;


    /*
     * 第二次遍历当前行，
     * 计算最终 RMSNorm 输出。
     *
     * 每个线程仍然处理与平方和阶段相同的列。
     */
    for (
        int64_t column = thread_id;
        column < hidden_size;
        column += blockDim.x
    ) {
        /*
         * 所有行共享同一组 weight。
         *
         * 最终公式：
         *
         *     output_i =
         *         x_i
         *         * inverse_rms
         *         * weight_i
         */
        row_output[column] =
            row_input[column]
            * inverse_rms
            * weight[column];
    }
}


/*
 * RMSNorm Warp CUDA Launcher。
 *
 * 这个函数运行在 CPU 端，
 * 负责配置和启动 rmsnorm_warp_kernel。
 *
 *
 * 后续我们会在 rmsnorm.cpp 中声明并调用：
 *
 *     rmsnorm_warp_cuda_launcher(...)
 *
 *
 * 参数：
 *
 *     x：
 *         CUDA FP32 输入张量。
 *
 *     weight：
 *         CUDA FP32 一维权重。
 *
 *     eps：
 *         数值稳定常数。
 *
 *
 * 返回值：
 *
 *     与 x 形状、dtype、device 相同的输出张量。
 */
torch::Tensor rmsnorm_warp_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    /*
     * 保证当前 CUDA Device
     * 与输入张量所在的 Device 一致。
     *
     * 当前 Jetson 通常只有 cuda:0，
     * 但保留 Device Guard 能让代码更加规范。
     */
    const c10::cuda::CUDAGuard device_guard(
        x.device()
    );


    /*
     * 创建输出张量。
     *
     * torch::empty_like(x) 会继承输入的：
     *
     *     - shape
     *     - dtype
     *     - device
     *     - memory format
     */
    torch::Tensor output =
        torch::empty_like(x);


    /*
     * 读取最后一个维度大小。
     *
     * RMSNorm 沿这个维度执行。
     */
    const int64_t hidden_size =
        x.size(-1);


    /*
     * 将输入的其他维度全部展平成 rows。
     *
     * 例如：
     *
     *     x.shape = [2, 8, 4096]
     *
     * 则：
     *
     *     rows = 2 * 8
     *          = 16
     */
    const int64_t rows =
        x.numel() / hidden_size;


    /*
     * 空输入处理。
     *
     * 对于：
     *
     *     x.shape = [0, hidden_size]
     *
     * rows 为 0，
     * 不需要启动 CUDA Kernel。
     */
    if (rows == 0) {
        return output;
    }


    /*
     * 每个 Block 固定使用 256 个线程。
     *
     * 256 个线程对应：
     *
     *     256 / 32 = 8 个 Warp
     *
     * 这是当前 Warp Reduce 版本的基础配置。
     */
    constexpr int threads_per_block =
        256;


    /*
     * 一个 Block 负责一行输入。
     *
     * 因此：
     *
     *     blocks = rows
     */
    const int64_t blocks =
        rows;


    /*
     * 当前 Block 中 Warp 的数量。
     *
     *     256 / 32 = 8
     */
    constexpr int warps_per_block =
        threads_per_block / kWarpSize;


    /*
     * 动态 Shared Memory 大小。
     *
     * 每个 Warp 只保存一个 float 类型的 Warp Sum。
     *
     * 当前为：
     *
     *     8 * sizeof(float)
     *     = 8 * 4
     *     = 32 bytes
     *
     * 对比 Naive 版本：
     *
     *     256 * sizeof(float)
     *     = 1024 bytes
     */
    constexpr std::size_t shared_memory_bytes =
        warps_per_block * sizeof(float);


    /*
     * 获取当前 PyTorch CUDA Stream。
     *
     * 自定义 Kernel 必须使用 PyTorch 当前 Stream，
     * 才能正确遵循前后 CUDA 操作的执行顺序。
     */
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();


    /*
     * 取得输入、权重和输出的底层 CUDA 内存指针。
     *
     * 后续会在 C++ 接口层保证：
     *
     *     x.dtype == float32
     *     weight.dtype == float32
     *
     * 所以这里使用：
     *
     *     data_ptr<float>()
     */
    const float* x_data =
        x.data_ptr<float>();


    const float* weight_data =
        weight.data_ptr<float>();


    float* output_data =
        output.data_ptr<float>();


    /*
     * 启动 Warp Reduce RMSNorm Kernel。
     *
     * CUDA 启动参数：
     *
     *     <<<grid, block, shared_memory, stream>>>
     *
     * grid：
     *
     *     rows 个 Block。
     *
     * block：
     *
     *     每个 Block 256 个线程。
     *
     * shared_memory：
     *
     *     每个 Block 32 字节动态 Shared Memory。
     *
     * stream：
     *
     *     当前 PyTorch CUDA Stream。
     */
    rmsnorm_warp_kernel<<<
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
     * 检查 Kernel 启动错误。
     *
     * 可以捕获：
     *
     *     - invalid configuration argument
     *     - invalid device function
     *     - illegal memory access
     *     - 其他 CUDA Runtime 错误
     *
     * 这里不会主动执行全局 cudaDeviceSynchronize()，
     * 因此不会破坏 PyTorch CUDA 的异步执行模式。
     */
    C10_CUDA_KERNEL_LAUNCH_CHECK();


    /*
     * 返回输出张量。
     */
    return output;
}
