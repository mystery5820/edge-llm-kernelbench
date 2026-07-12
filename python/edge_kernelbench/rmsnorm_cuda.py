"""
RMSNorm CUDA 扩展的 Python 加载与调用接口。

调用链如下：

    Python 用户代码
        ↓
    rmsnorm_cuda.py
        ↓
    rmsnorm.cpp
        ↓
    rmsnorm_kernel.cu
        ↓
    Jetson Orin GPU

这里使用 torch.utils.cpp_extension.load() 编译扩展。

第一次调用时会执行编译，后续调用会复用已经编译的扩展。
"""

from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load


# 保存已经加载的扩展模块。
#
# 初始值为 None，表示还没有编译或加载。
# 第一次调用 load_rmsnorm_cuda_extension() 后，
# 会将编译得到的 Python 扩展模块保存在这里。
_EXTENSION: Optional[ModuleType] = None


def _get_project_root() -> Path:
    """
    返回项目根目录。

    当前文件路径类似：

        edge-llm-kernelbench/
        └── python/
            └── edge_kernelbench/
                └── rmsnorm_cuda.py

    因此：

        Path(__file__).resolve().parents[0]
            -> python/edge_kernelbench

        parents[1]
            -> python

        parents[2]
            -> edge-llm-kernelbench
    """

    return Path(__file__).resolve().parents[2]


def load_rmsnorm_cuda_extension(
    verbose: bool = False,
) -> ModuleType:
    """
    编译并加载 RMSNorm CUDA 扩展。

    参数
    ----------
    verbose:
        是否显示完整编译日志。

        True：
            显示 nvcc、g++、ninja 等详细编译命令。

        False：
            只显示必要信息。

    返回
    ----
    ModuleType
        编译完成后的 Python 扩展模块。

        模块中包含：

            extension.forward(x, weight, eps)
    """

    global _EXTENSION

    # 如果扩展已经加载过，直接返回缓存。
    #
    # 避免在同一个 Python 进程内重复调用 load()。
    if _EXTENSION is not None:
        return _EXTENSION

    # CUDA Kernel 必须在 CUDA 可用的环境中运行。
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The RMSNorm CUDA extension requires a CUDA-capable device."
        )

    project_root = _get_project_root()

    # C++ 和 CUDA 源文件所在目录。
    kernel_directory = (
        project_root
        / "kernels"
        / "rmsnorm"
    )

    cpp_source = (
        kernel_directory
        / "rmsnorm.cpp"
    )

    cuda_source = (
        kernel_directory
        / "rmsnorm_kernel.cu"
    )

    # 编译产物存放目录。
    #
    # build/ 已经在项目的 .gitignore 中忽略，
    # 因此编译产生的中间文件不会被提交到 Git。
    build_directory = (
        project_root
        / "build"
        / "rmsnorm"
    )

    build_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 在真正编译前检查源文件是否存在。
    missing_sources = [
        source
        for source in (
            cpp_source,
            cuda_source,
        )
        if not source.is_file()
    ]

    if missing_sources:
        missing_text = "\n".join(
            str(source)
            for source in missing_sources
        )

        raise FileNotFoundError(
            "RMSNorm CUDA source files are missing:\n"
            f"{missing_text}"
        )

    # 使用 PyTorch JIT Extension 机制编译扩展。
    #
    # name:
    #     编译后 Python 模块的名称。
    #
    # sources:
    #     参与编译的 C++ 与 CUDA 源文件。
    #
    # extra_cflags:
    #     传给 g++ 的额外编译选项。
    #
    # extra_cuda_cflags:
    #     传给 nvcc 的额外编译选项。
    #
    # with_cuda=True:
    #     明确告诉 PyTorch 这是 CUDA 扩展。
    #
    # build_directory:
    #     指定编译中间文件与动态库的存放位置。
    _EXTENSION = load(
        name="edge_kernelbench_rmsnorm_cuda",
        sources=[
            str(cpp_source),
            str(cuda_source),
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


def rmsnorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    调用自定义 RMSNorm Naive CUDA Kernel。

    参数
    ----------
    x:
        CUDA FP32 输入张量。

        RMSNorm 沿最后一个维度执行。

    weight:
        一维 CUDA FP32 权重张量。

        形状必须为：

            [x.shape[-1]]

    eps:
        数值稳定常数，必须大于零。

    返回
    ----
    torch.Tensor
        与 x 具有相同形状、dtype 和 device 的输出张量。
    """

    extension = load_rmsnorm_cuda_extension()

    return extension.forward(
        x,
        weight,
        float(eps),
    )