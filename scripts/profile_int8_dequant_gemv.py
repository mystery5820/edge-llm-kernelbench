"""
Small Nsight profiling driver for INT8 Dequant-GEMV CUDA kernels.

The benchmark script is useful for end-to-end timing, but Nsight profiling
needs a quieter target that repeatedly launches one selected kernel after
the extension is already loaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIRECTORY = PROJECT_ROOT / "python"

if str(PYTHON_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIRECTORY),
    )


from edge_kernelbench.int8_dequant_gemv_cuda import (
    int8_dequant_gemv_cuda,
    int8_dequant_gemv_cuda_half2,
    int8_dequant_gemv_cuda_tiled,
    int8_dequant_gemv_cuda_vec4,
    int8_dequant_gemv_cuda_warp,
    int8_dequant_gemv_cuda_wide,
    load_int8_dequant_gemv_cuda_extension,
)


KERNEL_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "naive": int8_dequant_gemv_cuda,
    "warp": int8_dequant_gemv_cuda_warp,
    "tiled": int8_dequant_gemv_cuda_tiled,
    "vec4": int8_dequant_gemv_cuda_vec4,
    "wide": int8_dequant_gemv_cuda_wide,
    "half2": int8_dequant_gemv_cuda_half2,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one INT8 Dequant-GEMV CUDA implementation."
    )

    parser.add_argument(
        "--implementation",
        choices=sorted(KERNEL_FUNCTIONS),
        required=True,
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "fp32",
            "fp16",
        ],
        default="fp32",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--in-features",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--out-features",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--verbose-build",
        action="store_true",
    )

    arguments = parser.parse_args()

    if arguments.rows < 0:
        parser.error("--rows must be greater than or equal to zero")

    if arguments.in_features <= 0:
        parser.error("--in-features must be greater than zero")

    if arguments.out_features <= 0:
        parser.error("--out-features must be greater than zero")

    if arguments.warmup < 0:
        parser.error("--warmup must be greater than or equal to zero")

    if arguments.iterations <= 0:
        parser.error("--iterations must be greater than zero")

    if (
        arguments.implementation == "half2"
        and arguments.dtype != "fp16"
    ):
        parser.error("--implementation half2 requires --dtype fp16")

    return arguments


def main() -> None:
    arguments = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Profiling requires a CUDA-capable device."
        )

    torch.manual_seed(
        arguments.seed
    )

    dtype = (
        torch.float16
        if arguments.dtype == "fp16"
        else torch.float32
    )

    x = torch.randn(
        arguments.rows,
        arguments.in_features,
        device="cuda",
        dtype=dtype,
    )

    weight_int8 = torch.randint(
        -8,
        8,
        (
            arguments.out_features,
            arguments.in_features,
        ),
        device="cuda",
        dtype=torch.int8,
    )

    scale = torch.rand(
        arguments.out_features,
        device="cuda",
        dtype=torch.float32,
    )

    bias = torch.randn(
        arguments.out_features,
        device="cuda",
        dtype=torch.float32,
    )

    load_int8_dequant_gemv_cuda_extension(
        verbose=arguments.verbose_build,
    )

    kernel_function = KERNEL_FUNCTIONS[
        arguments.implementation
    ]

    with torch.inference_mode():
        for _ in range(arguments.warmup):
            output = kernel_function(
                x,
                weight_int8,
                scale,
                bias,
            )

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(
            enable_timing=True,
        )
        end_event = torch.cuda.Event(
            enable_timing=True,
        )

        start_event.record()

        for _ in range(arguments.iterations):
            output = kernel_function(
                x,
                weight_int8,
                scale,
                bias,
            )

        end_event.record()
        end_event.synchronize()

    elapsed_ms = start_event.elapsed_time(
        end_event
    )

    print(
        "implementation=",
        arguments.implementation,
        sep="",
    )
    print(
        "dtype=",
        arguments.dtype,
        sep="",
    )
    print(
        "shape=rows:",
        arguments.rows,
        ",in:",
        arguments.in_features,
        ",out:",
        arguments.out_features,
        sep="",
    )
    print(
        "iterations=",
        arguments.iterations,
        sep="",
    )
    print(
        "mean_ms=",
        f"{elapsed_ms / arguments.iterations:.6f}",
        sep="",
    )
    print(
        "output_checksum=",
        f"{float(output.float().sum().item()):.6f}",
        sep="",
    )


if __name__ == "__main__":
    main()
