"""
RMSNorm CUDA 扩展的 Python 加载与调用接口。

当前扩展包含两个 CUDA 实现：

1. rmsnorm_cuda()

   对应 Naive Shared Memory Reduce 版本。

2. rmsnorm_cuda_warp()

   对应 Warp Shuffle Reduce 版本。


完整调用关系：

    Python
        ↓
    rmsnorm_cuda.py
        ↓
    rmsnorm.cpp
        ├── rmsnorm_kernel.cu
        │       Naive Shared Memory Reduce
        │
        └── rmsnorm_warp_kernel.cu
                Warp Shuffle Reduce
        ↓
    Jetson Orin GPU


本文件使用：

    torch.utils.cpp_extension.load()

在运行时编译并加载 C++/CUDA 扩展。

第一次在新环境中运行时，会执行：

    g++
    nvcc
    ninja
    动态库链接

后续如果源文件没有变化，PyTorch 会复用已经生成的编译结果。
"""

from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from torch.utils.cpp_extension import load


# ---------------------------------------------------------------------------
# 扩展模块缓存
# ---------------------------------------------------------------------------

# 用于保存已经加载的 C++/CUDA 扩展模块。
#
# 初始值为 None，表示：
#
#     当前 Python 进程还没有加载扩展。
#
# 第一次调用 load_rmsnorm_cuda_extension() 后，
# 编译并加载得到的 Python 模块会存入 _EXTENSION。
#
# 同一个 Python 进程后续再次调用时，
# 可以直接返回缓存，不需要重复执行 load()。
_EXTENSION: Optional[ModuleType] = None


# ---------------------------------------------------------------------------
# 项目路径处理
# ---------------------------------------------------------------------------


def _get_project_root() -> Path:
    """
    返回 EdgeLLM-KernelBench 项目的根目录。

    当前文件通常位于：

        edge-llm-kernelbench/
        └── python/
            └── edge_kernelbench/
                └── rmsnorm_cuda.py

    Path(__file__).resolve() 得到当前文件的绝对路径。

    parents[0]：

        python/edge_kernelbench

    parents[1]：

        python

    parents[2]：

        edge-llm-kernelbench

    因此项目根目录为：

        Path(__file__).resolve().parents[2]
    """

    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CUDA 扩展编译与加载
# ---------------------------------------------------------------------------


def load_rmsnorm_cuda_extension(
    verbose: bool = False,
) -> ModuleType:
    """
    编译并加载 RMSNorm CUDA 扩展。

    当前扩展中包含：

        extension.forward(
            x,
            weight,
            eps,
        )

    以及：

        extension.forward_warp(
            x,
            weight,
            eps,
        )


    参数
    ----------
    verbose:
        是否显示完整的编译日志。

        False：

            只显示必要信息。

        True：

            显示 Ninja、g++、nvcc 和链接命令。


    返回
    ----
    ModuleType
        已经完成编译和加载的 Python 扩展模块。
    """

    global _EXTENSION

    # 如果当前 Python 进程已经加载过扩展，
    # 直接返回缓存。
    #
    # 这样可以避免同一个进程反复调用 load()。
    if _EXTENSION is not None:
        return _EXTENSION

    # 自定义 CUDA Kernel 必须运行在 CUDA 可用环境中。
    #
    # 如果在普通 CPU 机器上运行，
    # 这里会提前给出清晰错误，而不是等到编译或调用时报错。
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The RMSNorm CUDA extension requires a CUDA-capable device."
        )

    # 获取项目根目录。
    project_root = _get_project_root()

    # RMSNorm C++/CUDA 源文件所在目录：
    #
    #     edge-llm-kernelbench/kernels/rmsnorm
    kernel_directory = (
        project_root
        / "kernels"
        / "rmsnorm"
    )

    # PyBind11 和 PyTorch C++ 接口文件。
    #
    # 该文件负责：
    #
    #     - 输入检查；
    #     - 调用不同 CUDA Launcher；
    #     - 注册 Python 接口。
    cpp_source = (
        kernel_directory
        / "rmsnorm.cpp"
    )

    # Naive Shared Memory Reduce CUDA 实现。
    naive_cuda_source = (
        kernel_directory
        / "rmsnorm_kernel.cu"
    )

    # Warp Shuffle Reduce CUDA 实现。
    warp_cuda_source = (
        kernel_directory
        / "rmsnorm_warp_kernel.cu"
    )

    # 编译中间文件和最终动态库的存放目录。
    #
    # 目录结构类似：
    #
    #     build/rmsnorm/
    #         ├── build.ninja
    #         ├── rmsnorm.o
    #         ├── rmsnorm_kernel.cuda.o
    #         ├── rmsnorm_warp_kernel.cuda.o
    #         └── edge_kernelbench_rmsnorm_cuda.so
    #
    # 项目的 .gitignore 已经忽略 build/，
    # 所以这些编译产物不会进入 Git。
    build_directory = (
        project_root
        / "build"
        / "rmsnorm"
    )

    # 如果 build/rmsnorm 不存在，则自动创建。
    build_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 汇总当前扩展需要参与编译的全部源文件。
    sources = [
        cpp_source,
        naive_cuda_source,
        warp_cuda_source,
    ]

    # 在真正调用编译器之前，
    # 先检查全部源文件是否存在。
    #
    # 这样在文件名或路径写错时，
    # 可以得到更明确的错误信息。
    missing_sources = [
        source
        for source in sources
        if not source.is_file()
    ]

    if missing_sources:
        # 把所有缺失路径拼接成多行文本。
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
    # name：
    #
    #     最终加载到 Python 中的模块名称。
    #
    # sources：
    #
    #     需要编译的 C++ 和 CUDA 文件。
    #
    # extra_cflags：
    #
    #     传递给普通 C++ 编译器 g++ 的参数。
    #
    # extra_cuda_cflags：
    #
    #     传递给 CUDA 编译器 nvcc 的参数。
    #
    # with_cuda=True：
    #
    #     明确告诉 PyTorch 当前扩展包含 CUDA 源文件。
    #
    # build_directory：
    #
    #     指定中间文件和 .so 动态库的保存位置。
    #
    # verbose：
    #
    #     控制是否输出完整编译日志。
    _EXTENSION = load(
        name="edge_kernelbench_rmsnorm_cuda",
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


# ---------------------------------------------------------------------------
# Naive CUDA Python 接口
# ---------------------------------------------------------------------------


def rmsnorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    调用 RMSNorm FP32 Naive CUDA Kernel。

    该版本使用：

        Shared Memory 二叉树规约

    计算每一行的平方和。


    参数
    ----------
    x:
        CUDA FP32 输入张量。

        支持形状示例：

            [hidden_size]

            [rows, hidden_size]

            [batch_size, sequence_length, hidden_size]

        RMSNorm 始终沿最后一个维度执行。

    weight:
        CUDA FP32 一维权重张量。

        形状必须为：

            [x.shape[-1]]

    eps:
        数值稳定常数。

        必须严格大于零。


    返回
    ----
    torch.Tensor
        与 x 具有相同：

            - shape
            - dtype
            - device

        的 CUDA 输出张量。
    """

    # 获取已经加载的扩展模块。
    #
    # 第一次调用可能触发编译，
    # 后续调用会复用缓存。
    extension = load_rmsnorm_cuda_extension()

    # forward 对应 rmsnorm.cpp 中注册的：
    #
    #     rmsnorm_forward()
    #
    # 它最终会调用：
    #
    #     rmsnorm_cuda_launcher()
    #
    # 以及：
    #
    #     rmsnorm_naive_kernel()
    return extension.forward(
        x,
        weight,
        float(eps),
    )


# ---------------------------------------------------------------------------
# Warp Shuffle CUDA Python 接口
# ---------------------------------------------------------------------------


def rmsnorm_cuda_warp(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    调用 RMSNorm FP32 Warp Shuffle CUDA Kernel。

    该版本使用两级规约：

    第一级：

        每个 Warp 使用 __shfl_down_sync()
        在寄存器中完成 Warp 内平方和规约。

    第二级：

        每个 Warp 的 lane 0
        将结果写入 Shared Memory。

        然后由第一个 Warp
        汇总所有 Warp 的结果。


    相比 Naive 版本，该实现减少了：

        - Shared Memory 读写；
        - Block 级同步次数；
        - Shared Memory 使用量。


    参数
    ----------
    x:
        CUDA FP32 输入张量。

        RMSNorm 沿最后一个维度执行。

    weight:
        CUDA FP32 一维权重张量。

        元素数量必须等于：

            x.shape[-1]

    eps:
        数值稳定常数。

        必须严格大于零。


    返回
    ----
    torch.Tensor
        与 x 具有相同形状、dtype 和 device 的输出张量。
    """

    # 加载包含 Naive 和 Warp 两个实现的统一扩展模块。
    extension = load_rmsnorm_cuda_extension()

    # forward_warp 对应 rmsnorm.cpp 中注册的：
    #
    #     rmsnorm_warp_forward()
    #
    # 它最终会调用：
    #
    #     rmsnorm_warp_cuda_launcher()
    #
    # 以及：
    #
    #     rmsnorm_warp_kernel()
    return extension.forward_warp(
        x,
        weight,
        float(eps),
    )