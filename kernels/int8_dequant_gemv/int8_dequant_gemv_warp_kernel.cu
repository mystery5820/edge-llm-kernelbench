/*
 * INT8 Dequant-GEMV Warp-level CUDA Kernel。
 *
 * 一个 block 包含 8 个 warp。
 * 每个 warp 计算同一 row 的一个 out_feature。
 *
 * 相比 naive 版本：
 *
 * - naive：一个 block 计算一个 output[row, out_feature]
 * - warp：一个 block 计算 8 个 output[row, out_feature]
 *
 * 这样可以减少 block 数量，并用 warp shuffle 完成规约。
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr unsigned int kFullWarpMask = 0xffffffffu;


__device__ __forceinline__ float warp_reduce_sum_int8_gemv(
    float value
) {
    for (
        int offset = kWarpSize / 2;
        offset > 0;
        offset /= 2
    ) {
        value += __shfl_down_sync(
            kFullWarpMask,
            value,
            offset
        );
    }

    return value;
}


__global__ void int8_dequant_gemv_warp_kernel(
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
    const int thread_id =
        static_cast<int>(threadIdx.x);

    const int lane_id =
        thread_id & (kWarpSize - 1);

    const int warp_id =
        thread_id / kWarpSize;

    const int64_t row_id =
        static_cast<int64_t>(blockIdx.x);

    const int64_t out_feature_id =
        static_cast<int64_t>(blockIdx.y)
        * kWarpsPerBlock
        + static_cast<int64_t>(warp_id);

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
        int64_t column = lane_id;
        column < in_features;
        column += kWarpSize
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

    const float row_sum =
        warp_reduce_sum_int8_gemv(
            local_sum
        );

    if (lane_id == 0) {
        float value =
            row_sum
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


torch::Tensor int8_dequant_gemv_warp_cuda_launcher(
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

    const dim3 blocks(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(
            (
                out_features
                + kWarpsPerBlock
                - 1
            )
            / kWarpsPerBlock
        )
    );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    const bool has_bias =
        bias.numel() > 0;

    const float* bias_data =
        has_bias
            ? bias.data_ptr<float>()
            : nullptr;

    int8_dequant_gemv_warp_kernel<<<
        blocks,
        kThreadsPerBlock,
        0,
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
