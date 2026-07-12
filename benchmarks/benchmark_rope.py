"""
RoPE 性能 Benchmark。

比较：

1. PyTorch Reference
2. 自定义 FP32 Naive CUDA Kernel
3. 自定义 FP32 float4 CUDA Kernel
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


from edge_kernelbench.rope import rope_reference
from edge_kernelbench.rope_cuda import (
    load_rope_cuda_extension,
    rope_cuda,
    rope_cuda_float4,
)


@dataclass(frozen=True)
class BenchmarkCase:
    seq_len: int
    num_heads: int
    head_dim: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.seq_len,
            self.num_heads,
            self.head_dim,
        )

    @property
    def elements_per_tensor(self) -> int:
        return (
            self.seq_len
            * self.num_heads
            * self.head_dim
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
    seq_len: int
    num_heads: int
    head_dim: int
    elements_per_tensor: int
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
            "Benchmark PyTorch RoPE reference against "
            "the custom naive and float4 CUDA kernels."
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
) -> float:
    if not values:
        raise ValueError(
            "values must not be empty"
        )

    ordered_values = sorted(values)

    index = round(
        percentile
        * (len(ordered_values) - 1)
    )

    return ordered_values[index]


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
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmup_iterations: int,
    measurement_rounds: int,
    repeats_per_round: int,
) -> TimingStatistics:
    with torch.inference_mode():
        output: tuple[torch.Tensor, torch.Tensor] | None = None

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

            total_ms = start_event.elapsed_time(
                end_event
            )

            timings_ms.append(
                total_ms / repeats_per_round
            )

        if output is None:
            raise RuntimeError(
                "benchmark function did not produce an output"
            )

    return summarize_timings(
        timings_ms
    )


def verify_correctness(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[float, float, float, float]:
    with torch.inference_mode():
        reference_q, reference_k = rope_reference(
            q,
            k,
            cos,
            sin,
        )

        cuda_q, cuda_k = rope_cuda(
            q,
            k,
            cos,
            sin,
        )

        float4_q, float4_k = rope_cuda_float4(
            q,
            k,
            cos,
            sin,
        )

        torch.cuda.synchronize()

        q_maximum_error = (
            reference_q - cuda_q
        ).abs().max().item()

        k_maximum_error = (
            reference_k - cuda_k
        ).abs().max().item()

        float4_q_maximum_error = (
            reference_q - float4_q
        ).abs().max().item()

        float4_k_maximum_error = (
            reference_k - float4_k
        ).abs().max().item()

        torch.testing.assert_close(
            cuda_q,
            reference_q,
            rtol=1e-6,
            atol=1e-6,
        )

        torch.testing.assert_close(
            cuda_k,
            reference_k,
            rtol=1e-6,
            atol=1e-6,
        )

        torch.testing.assert_close(
            float4_q,
            reference_q,
            rtol=1e-6,
            atol=1e-6,
        )

        torch.testing.assert_close(
            float4_k,
            reference_k,
            rtol=1e-6,
            atol=1e-6,
        )

    return (
        q_maximum_error,
        k_maximum_error,
        float4_q_maximum_error,
        float4_k_maximum_error,
    )


def run_benchmark_case(
    case: BenchmarkCase,
    warmup_iterations: int,
    measurement_rounds: int,
    repeats_per_round: int,
) -> list[BenchmarkResult]:
    torch.manual_seed(2026)

    q = torch.randn(
        case.shape,
        device="cuda",
        dtype=torch.float32,
    )

    k = torch.randn_like(q)

    angles = torch.randn(
        case.seq_len,
        case.head_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )

    cos = torch.cos(
        angles
    ).contiguous()

    sin = torch.sin(
        angles
    ).contiguous()

    (
        q_maximum_error,
        k_maximum_error,
        float4_q_maximum_error,
        float4_k_maximum_error,
    ) = verify_correctness(
        q,
        k,
        cos,
        sin,
    )

    print(
        f"\nCase: seq_len={case.seq_len}, "
        f"num_heads={case.num_heads}, "
        f"head_dim={case.head_dim}, "
        f"elements_per_tensor={case.elements_per_tensor}"
    )

    print(
        "Q correctness maximum error: "
        f"{q_maximum_error:.8e}"
    )

    print(
        "K correctness maximum error: "
        f"{k_maximum_error:.8e}"
    )

    print(
        "Float4 Q correctness maximum error: "
        f"{float4_q_maximum_error:.8e}"
    )

    print(
        "Float4 K correctness maximum error: "
        f"{float4_k_maximum_error:.8e}"
    )

    reference_function = lambda: rope_reference(
        q,
        k,
        cos,
        sin,
    )

    cuda_function = lambda: rope_cuda(
        q,
        k,
        cos,
        sin,
    )

    float4_function = lambda: rope_cuda_float4(
        q,
        k,
        cos,
        sin,
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

    float4_statistics = benchmark_cuda_callable(
        function=float4_function,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
    )

    speedup = (
        reference_statistics.median_ms
        / cuda_statistics.median_ms
    )

    float4_speedup = (
        reference_statistics.median_ms
        / float4_statistics.median_ms
    )

    float4_speedup_vs_naive = (
        cuda_statistics.median_ms
        / float4_statistics.median_ms
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
        "CUDA Float4 median:       "
        f"{float4_statistics.median_ms:.6f} ms"
    )

    print(
        "CUDA Naive vs Reference:  "
        f"{speedup:.3f}x"
    )

    print(
        "CUDA Float4 vs Reference: "
        f"{float4_speedup:.3f}x"
    )

    print(
        "CUDA Float4 vs Naive:     "
        f"{float4_speedup_vs_naive:.3f}x"
    )

    return [
        BenchmarkResult(
            implementation="pytorch_reference",
            seq_len=case.seq_len,
            num_heads=case.num_heads,
            head_dim=case.head_dim,
            elements_per_tensor=case.elements_per_tensor,
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
            seq_len=case.seq_len,
            num_heads=case.num_heads,
            head_dim=case.head_dim,
            elements_per_tensor=case.elements_per_tensor,
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
            implementation="cuda_float4",
            seq_len=case.seq_len,
            num_heads=case.num_heads,
            head_dim=case.head_dim,
            elements_per_tensor=case.elements_per_tensor,
            warmup_iterations=warmup_iterations,
            measurement_rounds=measurement_rounds,
            repeats_per_round=repeats_per_round,
            mean_ms=float4_statistics.mean_ms,
            median_ms=float4_statistics.median_ms,
            minimum_ms=float4_statistics.minimum_ms,
            p95_ms=float4_statistics.p95_ms,
            speedup_vs_reference=float4_speedup,
        ),
    ]


def print_summary_table(
    results: list[BenchmarkResult],
) -> None:
    print("\n")
    print("=" * 114)
    print("RoPE Benchmark Summary")
    print("=" * 114)

    header = (
        f"{'implementation':<20}"
        f"{'seq':>8}"
        f"{'heads':>8}"
        f"{'head_dim':>10}"
        f"{'mean(ms)':>14}"
        f"{'median(ms)':>14}"
        f"{'min(ms)':>14}"
        f"{'p95(ms)':>14}"
        f"{'vs_ref':>12}"
    )

    print(header)
    print("-" * 114)

    for result in results:
        print(
            f"{result.implementation:<20}"
            f"{result.seq_len:>8}"
            f"{result.num_heads:>8}"
            f"{result.head_dim:>10}"
            f"{result.mean_ms:>14.6f}"
            f"{result.median_ms:>14.6f}"
            f"{result.minimum_ms:>14.6f}"
            f"{result.p95_ms:>14.6f}"
            f"{result.speedup_vs_reference:>11.3f}x"
        )

    print("=" * 114)


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
        / f"rope_float4_comparison_{timestamp}.csv"
    )

    fieldnames = [
        "timestamp",
        "device",
        "torch_version",
        "cuda_version",
        "implementation",
        "seq_len",
        "num_heads",
        "head_dim",
        "elements_per_tensor",
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
                    "seq_len": result.seq_len,
                    "num_heads": result.num_heads,
                    "head_dim": result.head_dim,
                    "elements_per_tensor": result.elements_per_tensor,
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
    print("EdgeLLM-KernelBench: RoPE PyTorch vs CUDA Naive vs Float4")
    print("=" * 72)
    print("PyTorch version: ", torch.__version__)
    print("CUDA version:    ", torch.version.cuda)
    print("CUDA device:     ", torch.cuda.get_device_name(0))
    print("Warmup calls:    ", arguments.warmup)
    print("Measure rounds:  ", arguments.rounds)
    print("Repeats/round:   ", arguments.repeats)

    print("\nLoading RoPE CUDA extension...")
    load_rope_cuda_extension(
        verbose=arguments.verbose_build
    )
    print("RoPE CUDA extension loaded.")

    benchmark_cases = [
        BenchmarkCase(
            seq_len=128,
            num_heads=8,
            head_dim=64,
        ),
        BenchmarkCase(
            seq_len=512,
            num_heads=8,
            head_dim=64,
        ),
        BenchmarkCase(
            seq_len=1024,
            num_heads=16,
            head_dim=64,
        ),
        BenchmarkCase(
            seq_len=2048,
            num_heads=16,
            head_dim=128,
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
