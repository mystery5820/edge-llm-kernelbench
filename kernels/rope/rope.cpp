/*
 * RoPE CUDA 扩展的 PyTorch C++ 接口层。
 *
 * 当前包含：
 *
 * - FP32 Naive CUDA 实现；
 * - FP32 float4 向量化 CUDA 实现。
 */

#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> rope_cuda_launcher(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
);


std::vector<torch::Tensor> rope_float4_cuda_launcher(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
);


void validate_rope_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin
) {
    TORCH_CHECK(
        q.is_cuda(),
        "q must be a CUDA tensor, but received device=",
        q.device()
    );

    TORCH_CHECK(
        k.is_cuda(),
        "k must be a CUDA tensor, but received device=",
        k.device()
    );

    TORCH_CHECK(
        cos.is_cuda(),
        "cos must be a CUDA tensor, but received device=",
        cos.device()
    );

    TORCH_CHECK(
        sin.is_cuda(),
        "sin must be a CUDA tensor, but received device=",
        sin.device()
    );

    TORCH_CHECK(
        q.device() == k.device()
            && q.device() == cos.device()
            && q.device() == sin.device(),
        "q, k, cos and sin must be on the same device"
    );

    TORCH_CHECK(
        q.sizes() == k.sizes(),
        "q and k must have the same shape: q.shape=",
        q.sizes(),
        ", k.shape=",
        k.sizes()
    );

    TORCH_CHECK(
        q.dim() >= 2,
        "q must have at least 2 dimensions"
    );

    TORCH_CHECK(
        q.scalar_type() == torch::kFloat32,
        "the RoPE CUDA kernel currently supports only float32 q, ",
        "but received dtype=",
        q.scalar_type()
    );

    TORCH_CHECK(
        k.scalar_type() == torch::kFloat32,
        "the RoPE CUDA kernel currently supports only float32 k, ",
        "but received dtype=",
        k.scalar_type()
    );

    TORCH_CHECK(
        cos.scalar_type() == torch::kFloat32,
        "the RoPE CUDA kernel currently supports only float32 cos, ",
        "but received dtype=",
        cos.scalar_type()
    );

    TORCH_CHECK(
        sin.scalar_type() == torch::kFloat32,
        "the RoPE CUDA kernel currently supports only float32 sin, ",
        "but received dtype=",
        sin.scalar_type()
    );

    TORCH_CHECK(
        q.is_contiguous(),
        "q must be contiguous"
    );

    TORCH_CHECK(
        k.is_contiguous(),
        "k must be contiguous"
    );

    TORCH_CHECK(
        cos.is_contiguous(),
        "cos must be contiguous"
    );

    TORCH_CHECK(
        sin.is_contiguous(),
        "sin must be contiguous"
    );

    const int64_t head_dim =
        q.size(-1);

    TORCH_CHECK(
        head_dim > 0,
        "the last dimension of q must be greater than zero"
    );

    TORCH_CHECK(
        head_dim % 2 == 0,
        "the last dimension of q must be even, but received ",
        head_dim
    );

    TORCH_CHECK(
        cos.dim() == 1 || cos.dim() == 2,
        "cos must be 1D or 2D, but received dim=",
        cos.dim()
    );

    TORCH_CHECK(
        sin.dim() == cos.dim(),
        "sin must have the same dim as cos: sin.dim=",
        sin.dim(),
        ", cos.dim=",
        cos.dim()
    );

    TORCH_CHECK(
        cos.sizes() == sin.sizes(),
        "cos and sin must have the same shape: cos.shape=",
        cos.sizes(),
        ", sin.shape=",
        sin.sizes()
    );

    const int64_t half_dim =
        head_dim / 2;

    TORCH_CHECK(
        cos.size(-1) == half_dim,
        "the last dimension of cos must be head_dim / 2: cos.size(-1)=",
        cos.size(-1),
        ", head_dim / 2=",
        half_dim
    );

    if (cos.dim() == 2) {
        const int64_t expected_seq_len =
            q.dim() == 2
                ? q.size(0)
                : q.size(-3);

        TORCH_CHECK(
            cos.size(0) == expected_seq_len,
            "cos.size(0) must match the sequence dimension: cos.size(0)=",
            cos.size(0),
            ", expected_seq_len=",
            expected_seq_len
        );
    }
}


std::vector<torch::Tensor> rope_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
) {
    validate_rope_inputs(
        q,
        k,
        cos,
        sin
    );

    return rope_cuda_launcher(
        q,
        k,
        cos,
        sin
    );
}


std::vector<torch::Tensor> rope_float4_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cos,
    torch::Tensor sin
) {
    validate_rope_inputs(
        q,
        k,
        cos,
        sin
    );

    return rope_float4_cuda_launcher(
        q,
        k,
        cos,
        sin
    );
}


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    module.def(
        "forward",
        &rope_forward,
        "RoPE naive CUDA forward"
    );

    module.def(
        "forward_float4",
        &rope_float4_forward,
        "RoPE float4 vectorized CUDA forward"
    );
}
