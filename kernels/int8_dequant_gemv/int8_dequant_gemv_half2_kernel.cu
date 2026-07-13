/*
 * INT8 Dequant-GEMV half2 CUDA Kernel。
 *
 * 基于 warp-level 映射：
 *
 * - 一个 block 8 个 warp；
 * - 每个 warp 计算一个 output[row, out_feature]；
 * - half2 快路径每个 lane 一次处理 2 个连续 column；
 * - x 使用 half2 读取；
 * - weight_int8 使用 char2 读取并转为 half2；
 * - half2 读取后转换为 FP32 乘加，保持与标量 FP16 路径一致的数值契约。
 *
 * 不满足对齐或 in_features % 2 != 0 时回退到标量 half 路径。
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


constexpr int kHalf2WarpSize = 32;
constexpr int kHalf2WarpsPerBlock = 8;
constexpr int kHalf2ThreadsPerBlock =
    kHalf2WarpSize * kHalf2WarpsPerBlock;
constexpr unsigned int kHalf2FullWarpMask = 0xffffffffu;


namespace {

bool is_aligned_half2_gemv(
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


__device__ __forceinline__ float warp_reduce_sum_int8_gemv_half2(
    float value
) {
    for (
        int offset = kHalf2WarpSize / 2;
        offset > 0;
        offset /= 2
    ) {
        value += __shfl_down_sync(
            kHalf2FullWarpMask,
            value,
            offset
        );
    }

    return value;
}


__device__ __forceinline__ float dot_half2_char2(
    half2 x_value,
    char2 weight_value
) {
    return (
        __low2float(x_value)
        * static_cast<float>(weight_value.x)
        + __high2float(x_value)
        * static_cast<float>(weight_value.y)
    );
}


__global__ void int8_dequant_gemv_half2_kernel(
    const half* __restrict__ x,
    const int8_t* __restrict__ weight_int8,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    half* __restrict__ output,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    bool has_bias,
    bool use_half2
) {
    const int thread_id =
        static_cast<int>(threadIdx.x);

    const int lane_id =
        thread_id & (kHalf2WarpSize - 1);

    const int warp_id =
        thread_id / kHalf2WarpSize;

    const int64_t row_id =
        static_cast<int64_t>(blockIdx.x);

    const int64_t out_feature_id =
        static_cast<int64_t>(blockIdx.y)
        * kHalf2WarpsPerBlock
        + static_cast<int64_t>(warp_id);

    if (
        row_id >= rows
        || out_feature_id >= out_features
    ) {
        return;
    }

    float local_sum =
        0.0f;

    if (use_half2) {
        const int64_t pairs_per_row =
            in_features / 2;

        const half2* x2 =
            reinterpret_cast<const half2*>(
                x
            );

        const char2* weight2 =
            reinterpret_cast<const char2*>(
                weight_int8
            );

        const int64_t x_pair_row_offset =
            row_id * pairs_per_row;

        const int64_t weight_pair_row_offset =
            out_feature_id * pairs_per_row;

        for (
            int64_t pair_column = lane_id;
            pair_column < pairs_per_row;
            pair_column += kHalf2WarpSize
        ) {
            const half2 x_value =
                x2[
                    x_pair_row_offset
                    + pair_column
                ];

            const char2 weight_value =
                weight2[
                    weight_pair_row_offset
                    + pair_column
                ];

            local_sum +=
                dot_half2_char2(
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
            column += kHalf2WarpSize
        ) {
            const float x_value =
                __half2float(
                    x[x_row_offset + column]
                );

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
        warp_reduce_sum_int8_gemv_half2(
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
        ] = __float2half_rn(
            value
        );
    }
}


torch::Tensor int8_dequant_gemv_half2_cuda_launcher(
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

    const bool use_half2 =
        in_features % 2 == 0
        && is_aligned_half2_gemv(
            x.data_ptr<at::Half>(),
            4
        )
        && is_aligned_half2_gemv(
            weight_int8.data_ptr<int8_t>(),
            2
        );

    const dim3 blocks(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(
            (
                out_features
                + kHalf2WarpsPerBlock
                - 1
            )
            / kHalf2WarpsPerBlock
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

    int8_dequant_gemv_half2_kernel<<<
        blocks,
        kHalf2ThreadsPerBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const half*>(
            x.data_ptr<at::Half>()
        ),
        weight_int8.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        bias_data,
        reinterpret_cast<half*>(
            output.data_ptr<at::Half>()
        ),
        rows,
        in_features,
        out_features,
        has_bias,
        use_half2
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
