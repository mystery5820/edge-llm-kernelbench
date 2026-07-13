"""
INT8 Dequant-GEMV 性能 Benchmark。
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
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


from edge_kernelbench.int8_dequant_gemv import int8_dequant_gemv_reference
from edge_kernelbench.int8_dequant_gemv_cuda import (
    int8_dequant_gemv_cuda,
    int8_dequant_gemv_cuda_tiled,
    int8_dequant_gemv_cuda_vec4,
    int8_dequant_gemv_cuda_warp,
    load_int8_dequant_gemv_cuda_extension,
)


@dataclass(frozen=True)
class BenchmarkCase:
    rows: int
    in_features: int
    out_features: int

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.rows,
            self.in_features,
        )


@dataclass(frozen=True)
class TimingStatistics:
    mean_ms: float
    median_ms: float
    minimum_ms: float
    p95_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    implementation: str
    rows: int
    in_features: int
    out_features: int
    warmup_iterations: int
    measurement_rounds: int
    repeats_per_round: int
    mean_ms: float
    median_ms: float
    minimum_ms: float
    p95_ms: float
    speedup_vs_reference: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PyTorch INT8 Dequant-GEMV reference "
            "against the custom naive, warp, tiled and vec4 CUDA kernels."
        )
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--verbose-build",
        action="store_true",
    )

    arguments = parser.parse_args()

    if arguments.warmup < 0:
        parser.error("--warmup must be greater than or equal to zero")

    if arguments.rounds <= 0:
        parser.error("--rounds must be greater than zero")

    if arguments.repeats <= 0:
        parser.error("--repeats must be greater than zero")

    return arguments


def calculate_percentile(
    values: list[float],
    percentile: float,
) -> tuple[float, float, float, float]:
    ordered = sorted(
        values
    )

    index = round(
        percentile
        * (len(ordered) - 1)
    )

    return ordered[index]


def summarize_timings(
    timings_ms: list[float],
) -> TimingStatistics:
    return TimingStatistics(
        mean_ms=statistics.mean(timings_ms),
        median_ms=statistics.median(timings_ms),
        minimum_ms=min(timings_ms),
        p95_ms=calculate_percentile(
            timings_ms,
            0.95,
        ),
    )


def benchmark_cuda_callable(
    function: Callable[[], torch.Tensor],
    warmup_iterations: int,
    measurement_rounds: int,
    repeats_per_round: int,
) -> TimingStatistics:
    with torch.inference_mode():
        output: torch.Tensor | None = None

        for _ in range(warmup_iterations):
            output = function()

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        timings_ms: list[float] = []

        for _ in range(measurement_rounds):
            start_event.record()

            for _ in range(repeats_per_round):
                output = function()

            end_event.record()
            end_event.synchronize()

            timings_ms.append(
                start_event.elapsed_time(
                    end_event
                )
                / repeats_per_round
            )

        if output is None:
            raise RuntimeError(
                "benchmark function did not produce an output"
            )

    return summarize_timings(
        timings_ms
    )


def verify_correctness(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> float:
    with torch.inference_mode():
        reference_output = int8_dequant_gemv_reference(
            x,
            weight_int8,
            scale,
            bias,
        )

        cuda_output = int8_dequant_gemv_cuda(
            x,
            weight_int8,
            scale,
            bias,
        )

        warp_output = int8_dequant_gemv_cuda_warp(
            x,
            weight_int8,
            scale,
            bias,
        )

        tiled_output = int8_dequant_gemv_cuda_tiled(
            x,
            weight_int8,
            scale,
            bias,
        )

        vec4_output = int8_dequant_gemv_cuda_vec4(
            x,
            weight_int8,
            scale,
            bias,
        )

        torch.cuda.synchronize()

        maximum_error = (
            reference_output - cuda_output
        ).abs().max().item()

        warp_maximum_error = (
            reference_output - warp_output
        ).abs().max().item()

        tiled_maximum_error = (
            reference_output - tiled_output
        ).abs().max().item()

        vec4_maximum_error = (
            reference_output - vec4_output
        ).abs().max().item()

        torch.testing.assert_close(
            cuda_output,
            reference_output,
            rtol=1e-4,
            atol=1e-4,
        )

        torch.testing.assert_close(
            warp_output,
            reference_output,
            rtol=1e-4,
            atol=1e-4,
        )

        torch.testing.assert_close(
            tiled_output,
            reference_output,
            rtol=1e-4,
            atol=1e-4,
        )

        torch.testing.assert_close(
            vec4_output,
            reference_output,
            rtol=1e-4,
            atol=1e-4,
        )

    return (
        maximum_error,
        warp_maximum_error,
        tiled_maximum_error,
        vec4_maximum_error,
    )


def run_benchmark_case(
    case: BenchmarkCase,
    warmup_iterations: int,
    measurement_rounds: int,
    repeats_per_round: int,
) -> list[BenchmarkResult]:
    torch.manual_seed(2026)

    x = torch.randn(
        case.shape,
        device="cuda",
        dtype=torch.float32,
    )

    weight_int8 = torch.randint(
        -8,
        8,
        (
            case.out_features,
            case.in_features,
        ),
        device="cuda",
        dtype=torch.int8,
    )

    scale = torch.rand(
        case.out_features,
        device="cuda",
        dtype=torch.float32,
    )

    bias = torch.randn(
        case.out_features,
        device="cuda",
        dtype=torch.float32,
    )

    (
        maximum_error,
        warp_maximum_error,
        tiled_maximum_error,
        vec4_maximum_error,
    ) = verify_correctness(
        x,
        weight_int8,
        scale,
        bias,
    )

    print(
        f"\nCase: rows={case.rows}, "
        f"in_features={case.in_features}, "
        f"out_features={case.out_features}"
    )

    print(
        "CUDA correctness maximum error: "
        f"{maximum_error:.8e}"
    )

    print(
        "Warp correctness maximum error: "
        f"{warp_maximum_error:.8e}"
    )

    print(
        "Tiled correctness maximum error: "
        f"{tiled_maximum_error:.8e}"
    )

    print(
        "Vec4 correctness maximum error: "
        f"{vec4_maximum_error:.8e}"
    )

    reference_function = lambda: int8_dequant_gemv_reference(
        x,
        weight_int8,
        scale,
        bias,
    )

    cuda_function = lambda: int8_dequant_gemv_cuda(
        x,
        weight_int8,
        scale,
        bias,
    )

    warp_function = lambda: int8_dequant_gemv_cuda_warp(
        x,
        weight_int8,
        scale,
        bias,
    )

    tiled_function = lambda: int8_dequant_gemv_cuda_tiled(
        x,
        weight_int8,
        scale,
        bias,
    )

    vec4_function = lambda: int8_dequant_gemv_cuda_vec4(
        x,
        weight_int8,
        scale,
        bias,
    )

    reference_statistics = benchmark_cuda_callable(
        function=reference_function,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
    )

    cuda_statistics = benchmark_cuda_callable(
        function=cuda_function,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
    )

    warp_statistics = benchmark_cuda_callable(
        function=warp_function,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
    )

    tiled_statistics = benchmark_cuda_callable(
        function=tiled_function,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
    )

    vec4_statistics = benchmark_cuda_callable(
        function=vec4_function,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
    )

    speedup = (
        reference_statistics.median_ms
        / cuda_statistics.median_ms
    )

    warp_speedup = (
        reference_statistics.median_ms
        / warp_statistics.median_ms
    )

    warp_speedup_vs_naive = (
        cuda_statistics.median_ms
        / warp_statistics.median_ms
    )

    tiled_speedup = (
        reference_statistics.median_ms
        / tiled_statistics.median_ms
    )

    tiled_speedup_vs_naive = (
        cuda_statistics.median_ms
        / tiled_statistics.median_ms
    )

    tiled_speedup_vs_warp = (
        warp_statistics.median_ms
        / tiled_statistics.median_ms
    )

    vec4_speedup = (
        reference_statistics.median_ms
        / vec4_statistics.median_ms
    )

    vec4_speedup_vs_naive = (
        cuda_statistics.median_ms
        / vec4_statistics.median_ms
    )

    vec4_speedup_vs_warp = (
        warp_statistics.median_ms
        / vec4_statistics.median_ms
    )

    print(
        "PyTorch Reference median: "
        f"{reference_statistics.median_ms:.6f} ms"
    )

    print(
        "CUDA Naive median:        "
        f"{cuda_statistics.median_ms:.6f} ms"
    )

    print(
        "CUDA Warp median:         "
        f"{warp_statistics.median_ms:.6f} ms"
    )

    print(
        "CUDA Tiled median:        "
        f"{tiled_statistics.median_ms:.6f} ms"
    )

    print(
        "CUDA Vec4 median:         "
        f"{vec4_statistics.median_ms:.6f} ms"
    )

    print(
        "CUDA Naive vs Reference:  "
        f"{speedup:.3f}x"
    )

    print(
        "CUDA Warp vs Reference:   "
        f"{warp_speedup:.3f}x"
    )

    print(
        "CUDA Tiled vs Reference:  "
        f"{tiled_speedup:.3f}x"
    )

    print(
        "CUDA Vec4 vs Reference:   "
        f"{vec4_speedup:.3f}x"
    )

    print(
        "CUDA Warp vs Naive:       "
        f"{warp_speedup_vs_naive:.3f}x"
    )

    print(
        "CUDA Tiled vs Naive:      "
        f"{tiled_speedup_vs_naive:.3f}x"
    )

    print(
        "CUDA Vec4 vs Naive:       "
        f"{vec4_speedup_vs_naive:.3f}x"
    )

    print(
        "CUDA Tiled vs Warp:       "
        f"{tiled_speedup_vs_warp:.3f}x"
    )

    print(
        "CUDA Vec4 vs Warp:        "
        f"{vec4_speedup_vs_warp:.3f}x"
    )

    return [
        BenchmarkResult(
            implementation="pytorch_reference",
            rows=case.rows,
            in_features=case.in_features,
            out_features=case.out_features,
            warmup_iterations=warmup_iterations,
            measurement_rounds=measurement_rounds,
            repeats_per_round=repeats_per_round,
            mean_ms=reference_statistics.mean_ms,
            median_ms=reference_statistics.median_ms,
            minimum_ms=reference_statistics.minimum_ms,
            p95_ms=reference_statistics.p95_ms,
            speedup_vs_reference=1.0,
        ),
        BenchmarkResult(
            implementation="cuda_naive",
            rows=case.rows,
            in_features=case.in_features,
            out_features=case.out_features,
            warmup_iterations=warmup_iterations,
            measurement_rounds=measurement_rounds,
            repeats_per_round=repeats_per_round,
            mean_ms=cuda_statistics.mean_ms,
            median_ms=cuda_statistics.median_ms,
            minimum_ms=cuda_statistics.minimum_ms,
            p95_ms=cuda_statistics.p95_ms,
            speedup_vs_reference=speedup,
        ),
        BenchmarkResult(
            implementation="cuda_warp",
            rows=case.rows,
            in_features=case.in_features,
            out_features=case.out_features,
            warmup_iterations=warmup_iterations,
            measurement_rounds=measurement_rounds,
            repeats_per_round=repeats_per_round,
            mean_ms=warp_statistics.mean_ms,
            median_ms=warp_statistics.median_ms,
            minimum_ms=warp_statistics.minimum_ms,
            p95_ms=warp_statistics.p95_ms,
            speedup_vs_reference=warp_speedup,
        ),
        BenchmarkResult(
            implementation="cuda_tiled",
            rows=case.rows,
            in_features=case.in_features,
            out_features=case.out_features,
            warmup_iterations=warmup_iterations,
            measurement_rounds=measurement_rounds,
            repeats_per_round=repeats_per_round,
            mean_ms=tiled_statistics.mean_ms,
            median_ms=tiled_statistics.median_ms,
            minimum_ms=tiled_statistics.minimum_ms,
            p95_ms=tiled_statistics.p95_ms,
            speedup_vs_reference=tiled_speedup,
        ),
        BenchmarkResult(
            implementation="cuda_vec4",
            rows=case.rows,
            in_features=case.in_features,
            out_features=case.out_features,
            warmup_iterations=warmup_iterations,
            measurement_rounds=measurement_rounds,
            repeats_per_round=repeats_per_round,
            mean_ms=vec4_statistics.mean_ms,
            median_ms=vec4_statistics.median_ms,
            minimum_ms=vec4_statistics.minimum_ms,
            p95_ms=vec4_statistics.p95_ms,
            speedup_vs_reference=vec4_speedup,
        ),
    ]


def print_summary_table(
    results: list[BenchmarkResult],
) -> None:
    print("\n")
    print("=" * 112)
    print("INT8 Dequant-GEMV Benchmark Summary")
    print("=" * 112)

    header = (
        f"{'implementation':<20}"
        f"{'rows':>8}"
        f"{'in':>10}"
        f"{'out':>10}"
        f"{'mean(ms)':>14}"
        f"{'median(ms)':>14}"
        f"{'min(ms)':>14}"
        f"{'p95(ms)':>14}"
        f"{'vs_ref':>12}"
    )

    print(header)
    print("-" * 112)

    for result in results:
        print(
            f"{result.implementation:<20}"
            f"{result.rows:>8}"
            f"{result.in_features:>10}"
            f"{result.out_features:>10}"
            f"{result.mean_ms:>14.6f}"
            f"{result.median_ms:>14.6f}"
            f"{result.minimum_ms:>14.6f}"
            f"{result.p95_ms:>14.6f}"
            f"{result.speedup_vs_reference:>11.3f}x"
        )

    print("=" * 112)


def save_results_to_csv(
    results: list[BenchmarkResult],
) -> Path:
    results_directory = (
        PROJECT_ROOT
        / "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        results_directory
        / f"int8_dequant_gemv_vec4_comparison_{timestamp}.csv"
    )

    fieldnames = [
        "timestamp",
        "device",
        "torch_version",
        "cuda_version",
        "implementation",
        "rows",
        "in_features",
        "out_features",
        "warmup_iterations",
        "measurement_rounds",
        "repeats_per_round",
        "mean_ms",
        "median_ms",
        "minimum_ms",
        "p95_ms",
        "speedup_vs_reference",
    ]

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "device": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda or "unknown",
                    "implementation": result.implementation,
                    "rows": result.rows,
                    "in_features": result.in_features,
                    "out_features": result.out_features,
                    "warmup_iterations": result.warmup_iterations,
                    "measurement_rounds": result.measurement_rounds,
                    "repeats_per_round": result.repeats_per_round,
                    "mean_ms": f"{result.mean_ms:.9f}",
                    "median_ms": f"{result.median_ms:.9f}",
                    "minimum_ms": f"{result.minimum_ms:.9f}",
                    "p95_ms": f"{result.p95_ms:.9f}",
                    "speedup_vs_reference": (
                        f"{result.speedup_vs_reference:.6f}"
                    ),
                }
            )

    return output_path


def main() -> None:
    arguments = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This benchmark must run on a CUDA-capable device."
        )

    print("=" * 72)
    print("EdgeLLM-KernelBench: INT8 Dequant-GEMV PyTorch vs CUDA Naive vs Warp vs Tiled vs Vec4")
    print("=" * 72)
    print("PyTorch version: ", torch.__version__)
    print("CUDA version:    ", torch.version.cuda)
    print("CUDA device:     ", torch.cuda.get_device_name(0))
    print("Warmup calls:    ", arguments.warmup)
    print("Measure rounds:  ", arguments.rounds)
    print("Repeats/round:   ", arguments.repeats)

    print("\nLoading INT8 Dequant-GEMV CUDA extension...")
    load_int8_dequant_gemv_cuda_extension(
        verbose=arguments.verbose_build
    )
    print("INT8 Dequant-GEMV CUDA extension loaded.")

    benchmark_cases = [
        BenchmarkCase(
            rows=1,
            in_features=1024,
            out_features=1024,
        ),
        BenchmarkCase(
            rows=1,
            in_features=2048,
            out_features=2048,
        ),
        BenchmarkCase(
            rows=4,
            in_features=2048,
            out_features=2048,
        ),
    ]

    all_results: list[BenchmarkResult] = []

    for case in benchmark_cases:
        all_results.extend(
            run_benchmark_case(
                case=case,
                warmup_iterations=arguments.warmup,
                measurement_rounds=arguments.rounds,
                repeats_per_round=arguments.repeats,
            )
        )

    print_summary_table(
        all_results
    )

    output_path = save_results_to_csv(
        all_results
    )

    print("\nBenchmark results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
