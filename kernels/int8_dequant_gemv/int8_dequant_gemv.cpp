/*
 * INT8 Dequant-GEMV CUDA 扩展的 C++ 接口层。
 */

#include <torch/extension.h>


torch::Tensor int8_dequant_gemv_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor bias
);


torch::Tensor int8_dequant_gemv_warp_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor bias
);


torch::Tensor int8_dequant_gemv_tiled_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor bias
);


void validate_int8_dequant_gemv_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_int8,
    const torch::Tensor& scale,
    const torch::Tensor& bias
) {
    TORCH_CHECK(
        x.is_cuda(),
        "x must be a CUDA tensor, but received device=",
        x.device()
    );

    TORCH_CHECK(
        weight_int8.is_cuda(),
        "weight_int8 must be a CUDA tensor, but received device=",
        weight_int8.device()
    );

    TORCH_CHECK(
        scale.is_cuda(),
        "scale must be a CUDA tensor, but received device=",
        scale.device()
    );

    const bool has_bias =
        bias.numel() > 0;

    if (has_bias) {
        TORCH_CHECK(
            bias.is_cuda(),
            "bias must be a CUDA tensor when provided, but received device=",
            bias.device()
        );
    }

    TORCH_CHECK(
        x.device() == weight_int8.device()
            && x.device() == scale.device()
            && (!has_bias || x.device() == bias.device()),
        "x, weight_int8, scale and bias must be on the same device"
    );

    TORCH_CHECK(
        x.dim() >= 1,
        "x must have at least one dimension"
    );

    TORCH_CHECK(
        weight_int8.dim() == 2,
        "weight_int8 must be a 2-dimensional tensor"
    );

    TORCH_CHECK(
        scale.dim() == 1,
        "scale must be a 1-dimensional tensor"
    );

    if (has_bias) {
        TORCH_CHECK(
            bias.dim() == 1,
            "bias must be a 1-dimensional tensor when provided"
        );
    }

    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        "the INT8 Dequant-GEMV CUDA kernel currently supports only float32 x, ",
        "but received dtype=",
        x.scalar_type()
    );

    TORCH_CHECK(
        weight_int8.scalar_type() == torch::kInt8,
        "weight_int8 must be torch.int8, but received dtype=",
        weight_int8.scalar_type()
    );

    TORCH_CHECK(
        scale.scalar_type() == torch::kFloat32,
        "scale must be torch.float32, but received dtype=",
        scale.scalar_type()
    );

    if (has_bias) {
        TORCH_CHECK(
            bias.scalar_type() == torch::kFloat32,
            "bias must be torch.float32 when provided, but received dtype=",
            bias.scalar_type()
        );
    }

    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous"
    );

    TORCH_CHECK(
        weight_int8.is_contiguous(),
        "weight_int8 must be contiguous"
    );

    TORCH_CHECK(
        scale.is_contiguous(),
        "scale must be contiguous"
    );

    if (has_bias) {
        TORCH_CHECK(
            bias.is_contiguous(),
            "bias must be contiguous"
        );
    }

    const int64_t out_features =
        weight_int8.size(0);

    const int64_t in_features =
        weight_int8.size(1);

    TORCH_CHECK(
        in_features > 0,
        "weight_int8.shape[1] must be greater than zero"
    );

    TORCH_CHECK(
        out_features > 0,
        "weight_int8.shape[0] must be greater than zero"
    );

    TORCH_CHECK(
        x.size(-1) == in_features,
        "the last dimension of x must match weight_int8.shape[1]: x.size(-1)=",
        x.size(-1),
        ", weight_int8.shape[1]=",
        in_features
    );

    TORCH_CHECK(
        scale.numel() == out_features,
        "scale.numel() must match weight_int8.shape[0]: scale.numel()=",
        scale.numel(),
        ", weight_int8.shape[0]=",
        out_features
    );

    if (has_bias) {
        TORCH_CHECK(
            bias.numel() == out_features,
            "bias.numel() must match weight_int8.shape[0]: bias.numel()=",
            bias.numel(),
            ", weight_int8.shape[0]=",
            out_features
        );
    }
}


torch::Tensor int8_dequant_gemv_forward(
    torch::Tensor x,
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor bias
) {
    validate_int8_dequant_gemv_inputs(
        x,
        weight_int8,
        scale,
        bias
    );

    return int8_dequant_gemv_cuda_launcher(
        x,
        weight_int8,
        scale,
        bias
    );
}


torch::Tensor int8_dequant_gemv_warp_forward(
    torch::Tensor x,
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor bias
) {
    validate_int8_dequant_gemv_inputs(
        x,
        weight_int8,
        scale,
        bias
    );

    return int8_dequant_gemv_warp_cuda_launcher(
        x,
        weight_int8,
        scale,
        bias
    );
}


torch::Tensor int8_dequant_gemv_tiled_forward(
    torch::Tensor x,
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor bias
) {
    validate_int8_dequant_gemv_inputs(
        x,
        weight_int8,
        scale,
        bias
    );

    return int8_dequant_gemv_tiled_cuda_launcher(
        x,
        weight_int8,
        scale,
        bias
    );
}


PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    module
) {
    module.def(
        "forward",
        &int8_dequant_gemv_forward,
        "INT8 Dequant-GEMV naive CUDA forward"
    );

    module.def(
        "forward_warp",
        &int8_dequant_gemv_warp_forward,
        "INT8 Dequant-GEMV warp-level CUDA forward"
    );

    module.def(
        "forward_tiled",
        &int8_dequant_gemv_tiled_forward,
        "INT8 Dequant-GEMV x-tile CUDA forward"
    );
}
