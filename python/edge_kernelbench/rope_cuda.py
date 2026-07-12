"""
RoPE CUDA 扩展的 Python 加载与调用接口。

当前扩展包含两个实现：

    rope_cuda(q, k, cos, sin)
    rope_cuda_float4(q, k, cos, sin)

分别对应 FP32 Naive CUDA Kernel 和 FP32 float4 向量化 CUDA Kernel。
"""

from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXTENSION: Optional[ModuleType] = None


def _get_project_root() -> Path:
    """
    返回项目根目录。
    """

    return Path(__file__).resolve().parents[2]


def load_rope_cuda_extension(
    verbose: bool = False,
) -> ModuleType:
    """
    编译并加载 RoPE CUDA 扩展。
    """

    global _EXTENSION

    if _EXTENSION is not None:
        return _EXTENSION

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The RoPE CUDA extension requires a CUDA-capable device."
        )

    project_root = _get_project_root()

    kernel_directory = (
        project_root
        / "kernels"
        / "rope"
    )

    cpp_source = (
        kernel_directory
        / "rope.cpp"
    )

    cuda_source = (
        kernel_directory
        / "rope_kernel.cu"
    )

    float4_cuda_source = (
        kernel_directory
        / "rope_float4_kernel.cu"
    )

    sources = [
        cpp_source,
        cuda_source,
        float4_cuda_source,
    ]

    missing_sources = [
        source
        for source in sources
        if not source.is_file()
    ]

    if missing_sources:
        missing_text = "\n".join(
            str(source)
            for source in missing_sources
        )

        raise FileNotFoundError(
            "RoPE CUDA source files are missing:\n"
            f"{missing_text}"
        )

    build_directory = (
        project_root
        / "build"
        / "rope"
    )

    build_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _EXTENSION = load(
        name="edge_kernelbench_rope_cuda",
        sources=[
            str(source)
            for source in sources
        ],
        extra_cflags=[
            "-O3",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
        ],
        with_cuda=True,
        build_directory=str(build_directory),
        verbose=verbose,
    )

    return _EXTENSION


def rope_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    调用 RoPE FP32 Naive CUDA Kernel。

    当前 CUDA 版本只支持 CUDA contiguous FP32 输入。
    """

    extension = load_rope_cuda_extension()

    q_output, k_output = extension.forward(
        q,
        k,
        cos,
        sin,
    )

    return (
        q_output,
        k_output,
    )


def rope_cuda_float4(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    调用 RoPE FP32 float4 向量化 CUDA Kernel。

    当 head_dim 不能被 4 整除，或 q/k/output 基址不满足 16 字节对齐时，
    CUDA kernel 会自动回退到标量路径。
    """

    extension = load_rope_cuda_extension()

    q_output, k_output = extension.forward_float4(
        q,
        k,
        cos,
        sin,
    )

    return (
        q_output,
        k_output,
    )
