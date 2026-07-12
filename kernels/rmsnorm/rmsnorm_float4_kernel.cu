/*
 * RMSNorm float4 向量化 CUDA Kernel。
 *
 * 本文件实现 RMSNorm 的第三个 CUDA 版本：
 *
 *     float4 Vectorized Load / Store 版本
 *
 * 它在 Warp Shuffle Reduce 版本基础上继续优化 Global Memory 访问。
 *
 *
 * 一、核心思路
 * -------------------------------------------------------------------------
 *
 * RMSNorm 需要对每一行执行两次访问：
 *
 * 1. 第一次读取输入 x，计算平方和；
 * 2. 第二次读取 x 和 weight，写出 output。
 *
 * Naive 与 Warp 版本都使用 float 标量访问。
 * 本版本在满足 16 字节对齐时，使用 float4 一次读取或写入 4 个 float。
 *
 * 对于 hidden_size 不是 4 的整数倍的情况：
 *
 *     前 hidden_size / 4 个 float4 使用向量化路径；
 *     剩余 0 到 3 个元素使用标量路径处理。
 *
 *
 * 二、对齐与回退
 * -------------------------------------------------------------------------
 *
 * float4 访问要求地址 16 字节对齐。
 *
 * 当前输入 x 虽然要求 contiguous，但仍可能因为 storage_offset 不是 4 的
 * 整数倍而导致首地址不是 16 字节对齐。
 *
 * 同时，当 hidden_size 不是 4 的整数倍时，即使第一行对齐，后续行首地址
 * 也可能因为行跨度不是 16 字节的整数倍而不对齐。
 *
 * 因此本 Kernel 对每一行单独检查：
 *
 *     row_input
 *     row_output
 *     weight
 *
 * 只有三者都 16 字节对齐时才走 float4 路径。
 * 否则该行回退到标量访问路径。
 *
 *
 * 三、线程组织方式
 * -------------------------------------------------------------------------
 *
 * 与前两个版本保持一致：
 *
 *     一个 CUDA Block 处理一行；
 *     每个 Block 固定 256 个线程；
 *     使用 Warp Shuffle 两级规约计算整行平方和。
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>


constexpr int kFloat4WarpSize = 32;

constexpr unsigned int kFloat4FullWarpMask =
    0xffffffffu;


/*
 * 判断一个地址是否满足 float4 访问所需的 16 字节对齐。
 *
 * CUDA 的 float4 包含 4 个 float：
 *
 *     4 * sizeof(float) = 16 bytes
 *
 * 如果地址低 4 位全为 0，则地址能被 16 整除。
 */
__device__ __forceinline__ bool is_float4_aligned(
    const void* pointer
) {
    const unsigned long long address =
        reinterpret_cast<unsigned long long>(
            pointer
        );

    return (address & 0x0full) == 0;
}


/*
 * Warp 内求和。
 *
 * 与 Warp Shuffle 版本保持相同规约方式，只是函数名带 float4，
 * 方便后续阅读时区分当前文件中的辅助函数。
 */
__device__ __forceinline__ float warp_reduce_sum_float4(
    float value
) {
    for (
        int offset = kFloat4WarpSize / 2;
        offset > 0;
        offset /= 2
    ) {
        value += __shfl_down_sync(
            kFloat4FullWarpMask,
            value,
            offset
        );
    }

    return value;
}


/*
 * 对一个 float4 中的四个分量求平方和。
 */
__device__ __forceinline__ float square_sum_float4(
    const float4 value
) {
    return (
        value.x * value.x
        + value.y * value.y
        + value.z * value.z
        + value.w * value.w
    );
}


/*
 * RMSNorm float4 CUDA Kernel。
 */
__global__ void rmsnorm_float4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t rows,
    int64_t hidden_size,
    float eps
) {
    /*
     * 保存每个 Warp 的局部平方和。
     *
     * 当前每个 Block 256 线程，即 8 个 Warp，因此 Launcher 会分配：
     *
     *     8 * sizeof(float)
     */
    extern __shared__ float shared_warp_sums[];

    /*
     * 整行 inverse_rms 只需计算一次。
     */
    __shared__ float shared_inverse_rms;

    const int thread_id =
        static_cast<int>(threadIdx.x);

    const int lane_id =
        thread_id & (kFloat4WarpSize - 1);

    const int warp_id =
        thread_id / kFloat4WarpSize;

    const int warp_count =
        (
            static_cast<int>(blockDim.x)
            + kFloat4WarpSize
            - 1
        )
        / kFloat4WarpSize;

    const int64_t row_id =
        static_cast<int64_t>(blockIdx.x);

    if (row_id >= rows) {
        return;
    }

    const int64_t row_offset =
        row_id * hidden_size;

    const float* row_input =
        x + row_offset;

    float* row_output =
        output + row_offset;

    /*
     * 每一行独立判断是否可以使用 float4。
     *
     * 这能正确处理：
     *
     *     hidden_size % 4 != 0
     *
     * 导致的后续行首地址不对齐问题。
     */
    const bool use_float4 =
        is_float4_aligned(row_input)
        && is_float4_aligned(row_output)
        && is_float4_aligned(weight);

    const int64_t vector_count =
        hidden_size / 4;

    const int64_t scalar_tail_begin =
        vector_count * 4;

    float local_square_sum =
        0.0f;

    if (use_float4) {
        const float4* row_input4 =
            reinterpret_cast<const float4*>(
                row_input
            );

        for (
            int64_t vector_column = thread_id;
            vector_column < vector_count;
            vector_column += blockDim.x
        ) {
            const float4 value =
                row_input4[vector_column];

            local_square_sum +=
                square_sum_float4(value);
        }

        /*
         * hidden_size 不能被 4 整除时，尾部 0 到 3 个元素使用标量处理。
         */
        for (
            int64_t column =
                scalar_tail_begin + thread_id;
            column < hidden_size;
            column += blockDim.x
        ) {
            const float value =
                row_input[column];

            local_square_sum +=
                value * value;
        }
    } else {
        /*
         * 当前行任一关键地址不满足 16 字节对齐时，
         * 回退到与 Warp 版本一致的标量读取路径。
         */
        for (
            int64_t column = thread_id;
            column < hidden_size;
            column += blockDim.x
        ) {
            const float value =
                row_input[column];

            local_square_sum +=
                value * value;
        }
    }

    const float warp_square_sum =
        warp_reduce_sum_float4(
            local_square_sum
        );

    if (lane_id == 0) {
        shared_warp_sums[warp_id] =
            warp_square_sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        float block_square_sum =
            lane_id < warp_count
                ? shared_warp_sums[lane_id]
                : 0.0f;

        block_square_sum =
            warp_reduce_sum_float4(
                block_square_sum
            );

        if (lane_id == 0) {
            const float mean_square =
                block_square_sum
                / static_cast<float>(
                    hidden_size
                );

            shared_inverse_rms =
                rsqrtf(
                    mean_square + eps
                );
        }
    }

    __syncthreads();

    const float inverse_rms =
        shared_inverse_rms;

    if (use_float4) {
        const float4* row_input4 =
            reinterpret_cast<const float4*>(
                row_input
            );

        const float4* weight4 =
            reinterpret_cast<const float4*>(
                weight
            );

        float4* row_output4 =
            reinterpret_cast<float4*>(
                row_output
            );

        for (
            int64_t vector_column = thread_id;
            vector_column < vector_count;
            vector_column += blockDim.x
        ) {
            const float4 input_value =
                row_input4[vector_column];

            const float4 weight_value =
                weight4[vector_column];

            float4 output_value;

            output_value.x =
                input_value.x
                * inverse_rms
                * weight_value.x;

            output_value.y =
                input_value.y
                * inverse_rms
                * weight_value.y;

            output_value.z =
                input_value.z
                * inverse_rms
                * weight_value.z;

            output_value.w =
                input_value.w
                * inverse_rms
                * weight_value.w;

            row_output4[vector_column] =
                output_value;
        }

        for (
            int64_t column =
                scalar_tail_begin + thread_id;
            column < hidden_size;
            column += blockDim.x
        ) {
            row_output[column] =
                row_input[column]
                * inverse_rms
                * weight[column];
        }
    } else {
        for (
            int64_t column = thread_id;
            column < hidden_size;
            column += blockDim.x
        ) {
            row_output[column] =
                row_input[column]
                * inverse_rms
                * weight[column];
        }
    }
}


/*
 * RMSNorm float4 CUDA Launcher。
 */
torch::Tensor rmsnorm_float4_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    const c10::cuda::CUDAGuard device_guard(
        x.device()
    );

    torch::Tensor output =
        torch::empty_like(x);

    const int64_t hidden_size =
        x.size(-1);

    const int64_t rows =
        x.numel() / hidden_size;

    if (rows == 0) {
        return output;
    }

    constexpr int threads_per_block =
        256;

    const int64_t blocks =
        rows;

    constexpr int warps_per_block =
        threads_per_block / kFloat4WarpSize;

    constexpr std::size_t shared_memory_bytes =
        warps_per_block * sizeof(float);

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    const float* x_data =
        x.data_ptr<float>();

    const float* weight_data =
        weight.data_ptr<float>();

    float* output_data =
        output.data_ptr<float>();

    rmsnorm_float4_kernel<<<
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

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
