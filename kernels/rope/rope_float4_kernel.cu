/*
 * RoPE float4 向量化 CUDA Kernel。
 *
 * 当前优化点：
 *
 * - 常见 head_dim=64/128 时，head_dim % 4 == 0；
 * - 每个线程使用一个 float4 处理两个 even/odd pair；
 * - q 和 k 同时处理；
 * - 不满足 16 字节对齐或 head_dim % 4 != 0 时回退到标量路径。
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


namespace {

bool is_aligned_16_bytes(
    const void* pointer
) {
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(
            pointer
        );

    return (address & 0x0f) == 0;
}

}  // namespace


__device__ __forceinline__ float2 apply_rope_pair_float4(
    float even_value,
    float odd_value,
    float cos_value,
    float sin_value
) {
    float2 output;

    output.x =
        even_value * cos_value
        - odd_value * sin_value;

    output.y =
        even_value * sin_value
        + odd_value * cos_value;

    return output;
}


__global__ void rope_float4_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ q_output,
    float* __restrict__ k_output,
    int64_t work_items,
    int64_t total_pairs,
    int64_t head_dim,
    int64_t half_dim,
    int64_t num_heads,
    int64_t seq_len,
    bool trig_has_sequence_dimension,
    bool use_float4
) {
    const int64_t thread_index =
        static_cast<int64_t>(blockIdx.x)
        * static_cast<int64_t>(blockDim.x)
        + static_cast<int64_t>(threadIdx.x);

    const int64_t stride =
        static_cast<int64_t>(gridDim.x)
        * static_cast<int64_t>(blockDim.x);

    if (use_float4) {
        const int64_t vectors_per_row =
            head_dim / 4;

        const float4* q4 =
            reinterpret_cast<const float4*>(
                q
            );

        const float4* k4 =
            reinterpret_cast<const float4*>(
                k
            );

        float4* q_output4 =
            reinterpret_cast<float4*>(
                q_output
            );

        float4* k_output4 =
            reinterpret_cast<float4*>(
                k_output
            );

        for (
            int64_t vector_index = thread_index;
            vector_index < work_items;
            vector_index += stride
        ) {
            const int64_t vector_in_row =
                vector_index % vectors_per_row;

            const int64_t row_id =
                vector_index / vectors_per_row;

            const int64_t first_pair_in_row =
                vector_in_row * 2;

            int64_t trig_base =
                first_pair_in_row;

            if (trig_has_sequence_dimension) {
                const int64_t sequence_id =
                    (row_id / num_heads)
                    % seq_len;

                trig_base =
                    sequence_id * half_dim
                    + first_pair_in_row;
            }

            const float cos0 =
                cos[trig_base];

            const float sin0 =
                sin[trig_base];

            const float cos1 =
                cos[trig_base + 1];

            const float sin1 =
                sin[trig_base + 1];

            const float4 q_value =
                q4[vector_index];

            const float4 k_value =
                k4[vector_index];

            const float2 q_pair0 =
                apply_rope_pair_float4(
                    q_value.x,
                    q_value.y,
                    cos0,
                    sin0
                );

            const float2 q_pair1 =
                apply_rope_pair_float4(
                    q_value.z,
                    q_value.w,
                    cos1,
                    sin1
                );

            const float2 k_pair0 =
                apply_rope_pair_float4(
                    k_value.x,
                    k_value.y,
                    cos0,
                    sin0
                );

            const float2 k_pair1 =
                apply_rope_pair_float4(
                    k_value.z,
                    k_value.w,
                    cos1,
                    sin1
                );

            float4 q_output_value;
            q_output_value.x = q_pair0.x;
            q_output_value.y = q_pair0.y;
            q_output_value.z = q_pair1.x;
            q_output_value.w = q_pair1.y;

            float4 k_output_value;
            k_output_value.x = k_pair0.x;
            k_output_value.y = k_pair0.y;
            k_output_value.z = k_pair1.x;
            k_output_value.w = k_pair1.y;

            q_output4[vector_index] =
                q_output_value;

            k_output4[vector_index] =
                k_output_value;
        }

        return;
    }

    for (
        int64_t pair_index = thread_index;
        pair_index < total_pairs;
        pair_index += stride
    ) {
        const int64_t pair_in_row =
            pair_index % half_dim;

        const int64_t row_id =
            pair_index / half_dim;

        const int64_t element_offset =
            row_id * head_dim
            + pair_in_row * 2;

        int64_t trig_offset =
            pair_in_row;

        if (trig_has_sequence_dimension) {
            const int64_t sequence_id =
                (row_id / num_heads)
                % seq_len;

            trig_offset =
                sequence_id * half_dim
                + pair_in_row;
        }

        const float cos_value =
            cos[trig_offset];

        const float sin_value =
            sin[trig_offset];

        const float q_even =
            q[element_offset];

        const float q_odd =
            q[element_offset + 1];

        const float k_even =
            k[element_offset];

        const float k_odd =
            k[element_offset + 1];

        q_output[element_offset] =
            q_even * cos_value
            - q_odd * sin_value;

        q_output[element_offset + 1] =
            q_even * sin_value
            + q_odd * cos_value;

        k_output[element_offset] =
            k_even * cos_value
            - k_odd * sin_value;

        k_output[element_offset + 1] =
            k_even * sin_value
            + k_odd * cos_value;
    }
}


std::vector<torch::Tensor> rope_float4_cuda_launcher(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
) {
    const c10::cuda::CUDAGuard device_guard(
        q.device()
    );

    torch::Tensor q_output =
        torch::empty_like(q);

    torch::Tensor k_output =
        torch::empty_like(k);

    if (q.numel() == 0) {
        return {
            q_output,
            k_output,
        };
    }

    const int64_t head_dim =
        q.size(-1);

    const int64_t half_dim =
        head_dim / 2;

    const int64_t total_pairs =
        q.numel() / 2;

    const int64_t num_heads =
        q.dim() >= 3
            ? q.size(-2)
            : 1;

    const int64_t seq_len =
        q.dim() == 2
            ? q.size(0)
            : q.size(-3);

    const bool trig_has_sequence_dimension =
        cos.dim() == 2;

    const bool use_float4 =
        head_dim % 4 == 0
        && is_aligned_16_bytes(
            q.data_ptr<float>()
        )
        && is_aligned_16_bytes(
            k.data_ptr<float>()
        )
        && is_aligned_16_bytes(
            q_output.data_ptr<float>()
        )
        && is_aligned_16_bytes(
            k_output.data_ptr<float>()
        );

    const int64_t work_items =
        use_float4
            ? q.numel() / 4
            : total_pairs;

    constexpr int threads_per_block =
        256;

    const int64_t blocks =
        (
            work_items
            + threads_per_block
            - 1
        )
        / threads_per_block;

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    rope_float4_kernel<<<
        static_cast<unsigned int>(blocks),
        threads_per_block,
        0,
        stream
    >>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        q_output.data_ptr<float>(),
        k_output.data_ptr<float>(),
        work_items,
        total_pairs,
        head_dim,
        half_dim,
        num_heads,
        seq_len,
        trig_has_sequence_dimension,
        use_float4
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        q_output,
        k_output,
    };
}
