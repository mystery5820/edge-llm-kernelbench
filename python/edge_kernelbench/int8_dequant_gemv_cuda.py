"""
INT8 Dequant-GEMV CUDA 扩展的 Python 加载与调用接口。
"""

from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXTENSION: Optional[ModuleType] = None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_int8_dequant_gemv_cuda_extension(
    verbose: bool = False,
) -> ModuleType:
    """
    编译并加载 INT8 Dequant-GEMV CUDA 扩展。
    """

    global _EXTENSION

    if _EXTENSION is not None:
        return _EXTENSION

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The INT8 Dequant-GEMV CUDA extension requires a CUDA-capable device."
        )

    project_root = _get_project_root()

    kernel_directory = (
        project_root
        / "kernels"
        / "int8_dequant_gemv"
    )

    cpp_source = (
        kernel_directory
        / "int8_dequant_gemv.cpp"
    )

    cuda_source = (
        kernel_directory
        / "int8_dequant_gemv_kernel.cu"
    )

    warp_cuda_source = (
        kernel_directory
        / "int8_dequant_gemv_warp_kernel.cu"
    )

    tiled_cuda_source = (
        kernel_directory
        / "int8_dequant_gemv_tiled_kernel.cu"
    )

    vec4_cuda_source = (
        kernel_directory
        / "int8_dequant_gemv_vec4_kernel.cu"
    )

    sources = [
        cpp_source,
        cuda_source,
        warp_cuda_source,
        tiled_cuda_source,
        vec4_cuda_source,
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
            "INT8 Dequant-GEMV CUDA source files are missing:\n"
            f"{missing_text}"
        )

    build_directory = (
        project_root
        / "build"
        / "int8_dequant_gemv"
    )

    build_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    _EXTENSION = load(
        name="edge_kernelbench_int8_dequant_gemv_cuda",
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


def int8_dequant_gemv_cuda(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    调用 INT8 Dequant-GEMV FP32 Naive CUDA Kernel。
    """

    extension = load_int8_dequant_gemv_cuda_extension()

    if bias is None:
        bias_for_extension = torch.empty(
            0,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        bias_for_extension = bias

    return extension.forward(
        x,
        weight_int8,
        scale,
        bias_for_extension,
    )


def int8_dequant_gemv_cuda_warp(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    调用 INT8 Dequant-GEMV warp-level CUDA Kernel。
    """

    extension = load_int8_dequant_gemv_cuda_extension()

    if bias is None:
        bias_for_extension = torch.empty(
            0,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        bias_for_extension = bias

    return extension.forward_warp(
        x,
        weight_int8,
        scale,
        bias_for_extension,
    )


def int8_dequant_gemv_cuda_tiled(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    调用 INT8 Dequant-GEMV x-tile shared-memory CUDA Kernel。
    """

    extension = load_int8_dequant_gemv_cuda_extension()

    if bias is None:
        bias_for_extension = torch.empty(
            0,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        bias_for_extension = bias

    return extension.forward_tiled(
        x,
        weight_int8,
        scale,
        bias_for_extension,
    )


def int8_dequant_gemv_cuda_vec4(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    调用 INT8 Dequant-GEMV vec4 CUDA Kernel。

    常见 in_features 为 4 的倍数且内存对齐时，x 使用 float4、
    weight_int8 使用 char4；否则 kernel 内部回退到标量 warp 路径。
    """

    extension = load_int8_dequant_gemv_cuda_extension()

    if bias is None:
        bias_for_extension = torch.empty(
            0,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        bias_for_extension = bias

    return extension.forward_vec4(
        x,
        weight_int8,
        scale,
        bias_for_extension,
    )
