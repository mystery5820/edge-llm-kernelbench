/*
 * INT8 Dequant-GEMV Tiled CUDA Kernel。
 *
 * 目标：
 *
 * - 保持 warp-level 版本的 8 个 warp / block 映射；
 * - 每个 block 仍然计算同一 row 的 8 个 out_feature；
 * - 将 x 的一段 tile 加载到 shared memory；
 * - 8 个 warp 复用同一段 x tile，减少重复 global load。
 *
 * 当前 tile 大小为 256 个 float。
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


constexpr int kTiledWarpSize = 32;
constexpr int kTiledWarpsPerBlock = 8;
constexpr int kTiledThreadsPerBlock =
    kTiledWarpSize * kTiledWarpsPerBlock;
constexpr int kXTileSize = 256;
constexpr unsigned int kTiledFullWarpMask = 0xffffffffu;


__device__ __forceinline__ float warp_reduce_sum_int8_gemv_tiled(
    float value
) {
    for (
        int offset = kTiledWarpSize / 2;
        offset > 0;
        offset /= 2
    ) {
        value += __shfl_down_sync(
            kTiledFullWarpMask,
            value,
            offset
        );
    }

    return value;
}


__global__ void int8_dequant_gemv_tiled_kernel(
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
    extern __shared__ float shared_x[];

    const int thread_id =
        static_cast<int>(threadIdx.x);

    const int lane_id =
        thread_id & (kTiledWarpSize - 1);

    const int warp_id =
        thread_id / kTiledWarpSize;

    const int64_t row_id =
        static_cast<int64_t>(blockIdx.x);

    const int64_t out_feature_id =
        static_cast<int64_t>(blockIdx.y)
        * kTiledWarpsPerBlock
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
        int64_t tile_start = 0;
        tile_start < in_features;
        tile_start += kXTileSize
    ) {
        const int64_t remaining =
            in_features - tile_start;

        const int tile_count =
            remaining < kXTileSize
                ? static_cast<int>(remaining)
                : kXTileSize;

        if (thread_id < tile_count) {
            shared_x[thread_id] =
                x[
                    x_row_offset
                    + tile_start
                    + thread_id
                ];
        }

        __syncthreads();

        for (
            int local_column = lane_id;
            local_column < tile_count;
            local_column += kTiledWarpSize
        ) {
            const float x_value =
                shared_x[local_column];

            const float weight_value =
                static_cast<float>(
                    weight_int8[
                        weight_row_offset
                        + tile_start
                        + local_column
                    ]
                );

            local_sum +=
                x_value * weight_value;
        }

        __syncthreads();
    }

    const float row_sum =
        warp_reduce_sum_int8_gemv_tiled(
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


torch::Tensor int8_dequant_gemv_tiled_cuda_launcher(
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
                + kTiledWarpsPerBlock
                - 1
            )
            / kTiledWarpsPerBlock
        )
    );

    constexpr std::size_t shared_memory_bytes =
        kXTileSize * sizeof(float);

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    const bool has_bias =
        bias.numel() > 0;

    const float* bias_data =
        has_bias
            ? bias.data_ptr<float>()
            : nullptr;

    int8_dequant_gemv_tiled_kernel<<<
        blocks,
        kTiledThreadsPerBlock,
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
