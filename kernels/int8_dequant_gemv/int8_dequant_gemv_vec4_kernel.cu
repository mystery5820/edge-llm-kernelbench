/*
 * INT8 Dequant-GEMV vec4 CUDA Kernel。
 *
 * 基于 warp-level 映射：
 *
 * - 一个 block 8 个 warp；
 * - 每个 warp 计算一个 output[row, out_feature]；
 * - 每个 lane 一次处理 4 个连续 column；
 * - x 使用 float4 读取；
 * - weight_int8 使用 char4 读取。
 *
 * 不满足对齐或 in_features % 4 != 0 时回退到标量 warp 路径。
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


constexpr int kVec4WarpSize = 32;
constexpr int kVec4WarpsPerBlock = 8;
constexpr int kVec4ThreadsPerBlock =
    kVec4WarpSize * kVec4WarpsPerBlock;
constexpr unsigned int kVec4FullWarpMask = 0xffffffffu;


namespace {

bool is_aligned(
    const void* pointer,
    std::uintptr_t alignment
) {
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(
            pointer
        );

    return (address & (alignment - 1)) == 0;
}

}  // namespace


__device__ __forceinline__ float warp_reduce_sum_int8_gemv_vec4(
    float value
) {
    for (
        int offset = kVec4WarpSize / 2;
        offset > 0;
        offset /= 2
    ) {
        value += __shfl_down_sync(
            kVec4FullWarpMask,
            value,
            offset
        );
    }

    return value;
}


__device__ __forceinline__ float dot_float4_char4(
    float4 x_value,
    char4 weight_value
) {
    return (
        x_value.x * static_cast<float>(weight_value.x)
        + x_value.y * static_cast<float>(weight_value.y)
        + x_value.z * static_cast<float>(weight_value.z)
        + x_value.w * static_cast<float>(weight_value.w)
    );
}


__global__ void int8_dequant_gemv_vec4_kernel(
    const float* __restrict__ x,
    const int8_t* __restrict__ weight_int8,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    bool has_bias,
    bool use_vec4
) {
    const int thread_id =
        static_cast<int>(threadIdx.x);

    const int lane_id =
        thread_id & (kVec4WarpSize - 1);

    const int warp_id =
        thread_id / kVec4WarpSize;

    const int64_t row_id =
        static_cast<int64_t>(blockIdx.x);

    const int64_t out_feature_id =
        static_cast<int64_t>(blockIdx.y)
        * kVec4WarpsPerBlock
        + static_cast<int64_t>(warp_id);

    if (
        row_id >= rows
        || out_feature_id >= out_features
    ) {
        return;
    }

    float local_sum =
        0.0f;

    if (use_vec4) {
        const int64_t vectors_per_row =
            in_features / 4;

        const float4* x4 =
            reinterpret_cast<const float4*>(
                x
            );

        const char4* weight4 =
            reinterpret_cast<const char4*>(
                weight_int8
            );

        const int64_t x_vector_row_offset =
            row_id * vectors_per_row;

        const int64_t weight_vector_row_offset =
            out_feature_id * vectors_per_row;

        for (
            int64_t vector_column = lane_id;
            vector_column < vectors_per_row;
            vector_column += kVec4WarpSize
        ) {
            const float4 x_value =
                x4[
                    x_vector_row_offset
                    + vector_column
                ];

            const char4 weight_value =
                weight4[
                    weight_vector_row_offset
                    + vector_column
                ];

            local_sum +=
                dot_float4_char4(
                    x_value,
                    weight_value
                );
        }
    } else {
        const int64_t x_row_offset =
            row_id * in_features;

        const int64_t weight_row_offset =
            out_feature_id * in_features;

        for (
            int64_t column = lane_id;
            column < in_features;
            column += kVec4WarpSize
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
    }

    const float row_sum =
        warp_reduce_sum_int8_gemv_vec4(
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


torch::Tensor int8_dequant_gemv_vec4_cuda_launcher(
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

    const bool use_vec4 =
        in_features % 4 == 0
        && is_aligned(
            x.data_ptr<float>(),
            16
        )
        && is_aligned(
            weight_int8.data_ptr<int8_t>(),
            4
        );

    const dim3 blocks(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(
            (
                out_features
                + kVec4WarpsPerBlock
                - 1
            )
            / kVec4WarpsPerBlock
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

    int8_dequant_gemv_vec4_kernel<<<
        blocks,
        kVec4ThreadsPerBlock,
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
        has_bias,
        use_vec4
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
