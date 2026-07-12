/*
 * RoPE Naive CUDA Kernel。
 *
 * 每个 CUDA 线程处理一个 even/odd 维度对，并同时计算 q 和 k。
 *
 * 当前实现范围：
 *
 * - FP32；
 * - q/k contiguous；
 * - q/k shape 相同；
 * - 支持：
 *
 *     [seq_len, head_dim]
 *     [seq_len, num_heads, head_dim]
 *     [batch_size, seq_len, num_heads, head_dim]
 *
 * - cos/sin 支持：
 *
 *     [head_dim / 2]
 *     [seq_len, head_dim / 2]
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


__global__ void rope_naive_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ q_output,
    float* __restrict__ k_output,
    int64_t total_pairs,
    int64_t head_dim,
    int64_t half_dim,
    int64_t num_heads,
    int64_t seq_len,
    bool trig_has_sequence_dimension
) {
    const int64_t thread_index =
        static_cast<int64_t>(blockIdx.x)
        * static_cast<int64_t>(blockDim.x)
        + static_cast<int64_t>(threadIdx.x);

    const int64_t stride =
        static_cast<int64_t>(gridDim.x)
        * static_cast<int64_t>(blockDim.x);

    for (
        int64_t pair_index = thread_index;
        pair_index < total_pairs;
        pair_index += stride
    ) {
        /*
         * pair_in_row 表示当前线程处理一行 head_dim 中第几个 even/odd 对。
         *
         * 例如 head_dim = 8，则 half_dim = 4：
         *
         *     pair_in_row = 0 -> dim 0 / 1
         *     pair_in_row = 1 -> dim 2 / 3
         *     pair_in_row = 2 -> dim 4 / 5
         *     pair_in_row = 3 -> dim 6 / 7
         */
        const int64_t pair_in_row =
            pair_index % half_dim;

        /*
         * row_id 是除最后 head_dim 外的展平行号。
         *
         * 对 [batch, seq, heads, head_dim]：
         *
         *     row_id = ((batch_id * seq_len + seq_id) * heads + head_id)
         */
        const int64_t row_id =
            pair_index / half_dim;

        const int64_t element_offset =
            row_id * head_dim
            + pair_in_row * 2;

        /*
         * cos/sin 为 1D 时，所有 token 共享同一组旋转参数。
         *
         * cos/sin 为 2D 时，需要从 row_id 反推出 seq_id：
         *
         *     seq_id = (row_id / num_heads) % seq_len
         *
         * 这里不需要显式传入 seq_len。
         * 因为 cos 的起始偏移只需要 seq_id * half_dim，
         * seq_id 通过取模前的结果即可覆盖所有 batch。
         */
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


std::vector<torch::Tensor> rope_cuda_launcher(
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

    constexpr int threads_per_block =
        256;

    const int64_t blocks =
        (
            total_pairs
            + threads_per_block
            - 1
        )
        / threads_per_block;

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    rope_naive_kernel<<<
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
        total_pairs,
        head_dim,
        half_dim,
        num_heads,
        seq_len,
        trig_has_sequence_dimension
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        q_output,
        k_output,
    };
}
