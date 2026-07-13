/*
 * INT8 Dequant-GEMV Naive CUDA Kernel。
 *
 * 一个 CUDA block 计算一个输出元素：
 *
 *     output[row, out_feature]
 *
 * block 内 256 个线程对 in_features 做 FP32 规约。
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


__global__ void int8_dequant_gemv_naive_kernel(
    const float* __restrict__ x,
    const int8_t* __restrict__ weight_int8,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    bool has_bias
) {
    extern __shared__ float shared_sum[];

    const int thread_id =
        static_cast<int>(threadIdx.x);

    const int64_t row_id =
        static_cast<int64_t>(blockIdx.x);

    const int64_t out_feature_id =
        static_cast<int64_t>(blockIdx.y);

    if (
        row_id >= rows
        || out_feature_id >= out_features
    ) {
        return;
    }

    const int64_t x_row_offset =
        row_id * in_features;

    const int64_t weight_row_offset =
        out_feature_id * in_features;

    float local_sum =
        0.0f;

    for (
        int64_t column = thread_id;
        column < in_features;
        column += blockDim.x
    ) {
        const float x_value =
            x[x_row_offset + column];

        const float weight_value =
            static_cast<float>(
                weight_int8[
                    weight_row_offset + column
                ]
            );

        local_sum +=
            x_value * weight_value;
    }

    shared_sum[thread_id] =
        local_sum;

    __syncthreads();

    for (
        unsigned int stride =
            blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (
            static_cast<unsigned int>(thread_id)
            < stride
        ) {
            shared_sum[thread_id] +=
                shared_sum[
                    thread_id + stride
                ];
        }

        __syncthreads();
    }

    if (thread_id == 0) {
        float value =
            shared_sum[0]
            * scale[out_feature_id];

        if (has_bias) {
            value +=
                bias[out_feature_id];
        }

        output[
            row_id * out_features
            + out_feature_id
        ] = value;
    }
}


torch::Tensor int8_dequant_gemv_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor bias
) {
    const c10::cuda::CUDAGuard device_guard(
        x.device()
    );

    const int64_t in_features =
        weight_int8.size(1);

    const int64_t out_features =
        weight_int8.size(0);

    const int64_t rows =
        x.numel() / in_features;

    std::vector<int64_t> output_sizes =
        x.sizes().vec();

    output_sizes.back() =
        out_features;

    torch::Tensor output =
        torch::empty(
            output_sizes,
            x.options()
        );

    if (rows == 0) {
        return output;
    }

    constexpr int threads_per_block =
        256;

    const dim3 blocks(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(out_features)
    );

    constexpr std::size_t shared_memory_bytes =
        threads_per_block * sizeof(float);

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    const bool has_bias =
        bias.numel() > 0;

    const float* bias_data =
        has_bias
            ? bias.data_ptr<float>()
            : nullptr;

    int8_dequant_gemv_naive_kernel<<<
        blocks,
        threads_per_block,
        shared_memory_bytes,
        stream
    >>>(
        x.data_ptr<float>(),
        weight_int8.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        bias_data,
        output.data_ptr<float>(),
        rows,
        in_features,
        out_features,
        has_bias
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
