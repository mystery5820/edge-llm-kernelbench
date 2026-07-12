"""
RMSNorm 性能基线 Benchmark。

本脚本用于比较两种 RMSNorm 实现：

1. PyTorch Reference
2. 自定义 FP32 Naive CUDA Kernel

测试流程：

    创建输入
        ↓
    检查数值正确性
        ↓
    CUDA 预热
        ↓
    使用 CUDA Event 测量延迟
        ↓
    统计 mean / median / min / P95
        ↓
    计算自定义 Kernel 相对 PyTorch 的加速比
        ↓
    将结果写入 CSV 文件

运行方式：

    MAX_JOBS=2 PYTHONPATH=python \
    python benchmarks/benchmark_rmsnorm.py

也可以修改测试参数：

    MAX_JOBS=2 PYTHONPATH=python \
    python benchmarks/benchmark_rmsnorm.py \
        --warmup 20 \
        --rounds 30 \
        --repeats 50
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


# ---------------------------------------------------------------------------
# 项目路径处理
# ---------------------------------------------------------------------------

# 当前文件位于：
#
#     edge-llm-kernelbench/
#     └── benchmarks/
#         └── benchmark_rmsnorm.py
#
# 因此 parents[1] 就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 项目 Python 包位于：
#
#     edge-llm-kernelbench/python
PYTHON_DIRECTORY = PROJECT_ROOT / "python"

# 将 python/ 目录加入模块搜索路径。
#
# 这样即使没有安装 edge_kernelbench 包，
# 也可以直接从源码目录导入。
if str(PYTHON_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIRECTORY),
    )


from edge_kernelbench.rmsnorm import rmsnorm_reference
from edge_kernelbench.rmsnorm_cuda import (
    load_rmsnorm_cuda_extension,
    rmsnorm_cuda,
)


# ---------------------------------------------------------------------------
# Benchmark 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    """
    一组 RMSNorm Benchmark 输入配置。

    RMSNorm 会沿最后一个维度进行归一化。

    在 CUDA Kernel 内部，输入统一被视为：

        [rows, hidden_size]

    rows:
        需要进行 RMSNorm 的向量数量。

    hidden_size:
        每一个向量的元素数量。
    """

    rows: int
    hidden_size: int

    @property
    def shape(self) -> tuple[int, int]:
        """
        返回输入张量形状。
        """

        return (
            self.rows,
            self.hidden_size,
        )

    @property
    def numel(self) -> int:
        """
        返回输入张量元素总数。
        """

        return (
            self.rows
            * self.hidden_size
        )


@dataclass(frozen=True)
class TimingStatistics:
    """
    一种实现的延迟统计结果。

    所有时间单位均为毫秒。
    """

    mean_ms: float
    median_ms: float
    minimum_ms: float
    p95_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    """
    一条完整的 Benchmark 结果。
    """

    implementation: str
    rows: int
    hidden_size: int
    elements: int
    warmup_iterations: int
    measurement_rounds: int
    repeats_per_round: int
    mean_ms: float
    median_ms: float
    minimum_ms: float
    p95_ms: float
    speedup_vs_reference: float


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PyTorch RMSNorm reference "
            "against the custom naive CUDA kernel."
        )
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help=(
            "Number of warmup calls before measurement. "
            "Default: 20"
        ),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=30,
        help=(
            "Number of measurement rounds. "
            "Default: 30"
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help=(
            "Number of operator calls inside each measurement round. "
            "Default: 50"
        ),
    )

    parser.add_argument(
        "--verbose-build",
        action="store_true",
        help=(
            "Print complete C++ and CUDA extension build logs."
        ),
    )

    arguments = parser.parse_args()

    if arguments.warmup < 0:
        parser.error("--warmup must be greater than or equal to zero")

    if arguments.rounds <= 0:
        parser.error("--rounds must be greater than zero")

    if arguments.repeats <= 0:
        parser.error("--repeats must be greater than zero")

    return arguments


# ---------------------------------------------------------------------------
# 延迟统计
# ---------------------------------------------------------------------------


def calculate_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """
    计算简单百分位数。

    参数
    ----------
    values:
        延迟样本。

    percentile:
        百分位，范围为 0 到 1。

        例如：

            0.95 表示 P95。
    """

    if not values:
        raise ValueError(
            "values must not be empty"
        )

    if not 0.0 <= percentile <= 1.0:
        raise ValueError(
            "percentile must be between 0 and 1"
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
    """
    将多轮延迟样本汇总为统计数据。
    """

    if not timings_ms:
        raise ValueError(
            "timings_ms must not be empty"
        )

    return TimingStatistics(
        mean_ms=statistics.mean(timings_ms),
        median_ms=statistics.median(timings_ms),
        minimum_ms=min(timings_ms),
        p95_ms=calculate_percentile(
            timings_ms,
            0.95,
        ),
    )


# ---------------------------------------------------------------------------
# CUDA Benchmark 核心
# ---------------------------------------------------------------------------


def benchmark_cuda_callable(
    function: Callable[[], torch.Tensor],
    warmup_iterations: int,
    measurement_rounds: int,
    repeats_per_round: int,
) -> TimingStatistics:
    """
    使用 CUDA Event 测量一个 CUDA 函数的执行延迟。

    为了减少 Event 记录本身对微小 Kernel 的影响，
    每一轮会连续执行 repeats_per_round 次函数调用。

    一轮的单次平均延迟为：

        elapsed_time / repeats_per_round
    """

    # Benchmark 只测试前向计算，
    # 不需要构建 Autograd 计算图。
    with torch.inference_mode():
        # ---------------------------------------------------------------
        # CUDA 预热
        # ---------------------------------------------------------------
        #
        # 预热用于排除：
        #
        # - 第一次 CUDA Context 初始化；
        # - 第一次内存分配；
        # - 第一次扩展加载；
        # - 缓存未命中；
        # - GPU 尚未进入稳定工作状态。
        output: torch.Tensor | None = None

        for _ in range(warmup_iterations):
            output = function()

        torch.cuda.synchronize()

        # ---------------------------------------------------------------
        # 创建 CUDA Event
        # ---------------------------------------------------------------
        #
        # CUDA Event 在 GPU 时间线上记录时间戳，
        # 比 Python time.perf_counter() 更适合测量异步 CUDA 操作。
        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        round_timings_ms: list[float] = []

        # ---------------------------------------------------------------
        # 正式测量
        # ---------------------------------------------------------------
        for _ in range(measurement_rounds):
            start_event.record()

            for _ in range(repeats_per_round):
                output = function()

            end_event.record()

            # 等待当前测量轮真正执行完毕。
            end_event.synchronize()

            # elapsed_time 返回毫秒。
            total_elapsed_ms = start_event.elapsed_time(
                end_event
            )

            average_call_ms = (
                total_elapsed_ms
                / repeats_per_round
            )

            round_timings_ms.append(
                average_call_ms
            )

        # 保留 output 引用直到测量完成，
        # 避免最后一个返回值过早离开作用域。
        if output is None:
            raise RuntimeError(
                "benchmark function did not produce an output"
            )

    return summarize_timings(
        round_timings_ms
    )


# ---------------------------------------------------------------------------
# 正确性检查
# ---------------------------------------------------------------------------


def verify_correctness(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> float:
    """
    在性能测试前检查自定义 CUDA Kernel 的数值正确性。

    返回最大绝对误差。
    """

    with torch.inference_mode():
        expected = rmsnorm_reference(
            x,
            weight,
            eps,
        )

        actual = rmsnorm_cuda(
            x,
            weight,
            eps,
        )

        torch.cuda.synchronize()

        maximum_absolute_error = (
            expected - actual
        ).abs().max().item()

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    return maximum_absolute_error


# ---------------------------------------------------------------------------
# 单个输入配置测试
# ---------------------------------------------------------------------------


def run_benchmark_case(
    case: BenchmarkCase,
    warmup_iterations: int,
    measurement_rounds: int,
    repeats_per_round: int,
    eps: float,
) -> list[BenchmarkResult]:
    """
    测试一组 rows 和 hidden_size。
    """

    torch.manual_seed(2026)

    x = torch.randn(
        case.shape,
        device="cuda",
        dtype=torch.float32,
    )

    weight = torch.randn(
        case.hidden_size,
        device="cuda",
        dtype=torch.float32,
    )

    maximum_error = verify_correctness(
        x,
        weight,
        eps,
    )

    print(
        f"\nCase: rows={case.rows}, "
        f"hidden_size={case.hidden_size}, "
        f"elements={case.numel}"
    )

    print(
        "Correctness maximum absolute error: "
        f"{maximum_error:.8e}"
    )

    # 使用 lambda 固定输入参数，
    # 让 benchmark_cuda_callable 只关心一次函数调用。
    reference_function = lambda: rmsnorm_reference(
        x,
        weight,
        eps,
    )

    cuda_naive_function = lambda: rmsnorm_cuda(
        x,
        weight,
        eps,
    )

    reference_statistics = benchmark_cuda_callable(
        function=reference_function,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
    )

    cuda_statistics = benchmark_cuda_callable(
        function=cuda_naive_function,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
    )

    # 加速比采用 median 延迟计算：
    #
    #     speedup = reference_time / custom_time
    #
    # speedup > 1：
    #     自定义 Kernel 更快。
    #
    # speedup < 1：
    #     自定义 Kernel 更慢。
    speedup = (
        reference_statistics.median_ms
        / cuda_statistics.median_ms
    )

    reference_result = BenchmarkResult(
        implementation="pytorch_reference",
        rows=case.rows,
        hidden_size=case.hidden_size,
        elements=case.numel,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
        mean_ms=reference_statistics.mean_ms,
        median_ms=reference_statistics.median_ms,
        minimum_ms=reference_statistics.minimum_ms,
        p95_ms=reference_statistics.p95_ms,
        speedup_vs_reference=1.0,
    )

    cuda_result = BenchmarkResult(
        implementation="cuda_naive",
        rows=case.rows,
        hidden_size=case.hidden_size,
        elements=case.numel,
        warmup_iterations=warmup_iterations,
        measurement_rounds=measurement_rounds,
        repeats_per_round=repeats_per_round,
        mean_ms=cuda_statistics.mean_ms,
        median_ms=cuda_statistics.median_ms,
        minimum_ms=cuda_statistics.minimum_ms,
        p95_ms=cuda_statistics.p95_ms,
        speedup_vs_reference=speedup,
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
        "CUDA Naive speedup:       "
        f"{speedup:.3f}x"
    )

    return [
        reference_result,
        cuda_result,
    ]


# ---------------------------------------------------------------------------
# 结果展示与保存
# ---------------------------------------------------------------------------


def print_summary_table(
    results: list[BenchmarkResult],
) -> None:
    """
    在终端打印汇总表。
    """

    print("\n")
    print("=" * 112)
    print("RMSNorm Benchmark Summary")
    print("=" * 112)

    header = (
        f"{'implementation':<20}"
        f"{'rows':>8}"
        f"{'hidden':>10}"
        f"{'mean(ms)':>14}"
        f"{'median(ms)':>14}"
        f"{'min(ms)':>14}"
        f"{'p95(ms)':>14}"
        f"{'speedup':>12}"
    )

    print(header)
    print("-" * 112)

    for result in results:
        row = (
            f"{result.implementation:<20}"
            f"{result.rows:>8}"
            f"{result.hidden_size:>10}"
            f"{result.mean_ms:>14.6f}"
            f"{result.median_ms:>14.6f}"
            f"{result.minimum_ms:>14.6f}"
            f"{result.p95_ms:>14.6f}"
            f"{result.speedup_vs_reference:>11.3f}x"
        )

        print(row)

    print("=" * 112)


def save_results_to_csv(
    results: list[BenchmarkResult],
) -> Path:
    """
    将结果保存到带时间戳的 CSV 文件。
    """

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
        / f"rmsnorm_baseline_{timestamp}.csv"
    )

    device_name = torch.cuda.get_device_name(0)
    torch_version = torch.__version__
    cuda_version = torch.version.cuda or "unknown"

    fieldnames = [
        "timestamp",
        "device",
        "torch_version",
        "cuda_version",
        "implementation",
        "rows",
        "hidden_size",
        "elements",
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
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "device": device_name,
                    "torch_version": torch_version,
                    "cuda_version": cuda_version,
                    "implementation": result.implementation,
                    "rows": result.rows,
                    "hidden_size": result.hidden_size,
                    "elements": result.elements,
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


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Benchmark 主入口。
    """

    arguments = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This benchmark must run on a CUDA-capable device."
        )

    print("=" * 72)
    print("EdgeLLM-KernelBench: RMSNorm Baseline")
    print("=" * 72)

    print(
        "PyTorch version: ",
        torch.__version__,
    )

    print(
        "CUDA version:    ",
        torch.version.cuda,
    )

    print(
        "CUDA device:     ",
        torch.cuda.get_device_name(0),
    )

    print(
        "Warmup calls:    ",
        arguments.warmup,
    )

    print(
        "Measure rounds:  ",
        arguments.rounds,
    )

    print(
        "Repeats/round:   ",
        arguments.repeats,
    )

    # 在正式 Benchmark 前显式加载扩展。
    #
    # 这样扩展编译时间不会进入性能测量结果。
    print("\nLoading RMSNorm CUDA extension...")

    load_rmsnorm_cuda_extension(
        verbose=arguments.verbose_build
    )

    print("RMSNorm CUDA extension loaded.")

    # 当前第一批性能基线。
    #
    # rows=1：
    #     模拟单 Token 或极小批量场景。
    #
    # rows=16、128：
    #     模拟多个 Token 同时进行归一化的场景。
    benchmark_cases = [
        BenchmarkCase(
            rows=1,
            hidden_size=1024,
        ),
        BenchmarkCase(
            rows=1,
            hidden_size=4096,
        ),
        BenchmarkCase(
            rows=16,
            hidden_size=4096,
        ),
        BenchmarkCase(
            rows=128,
            hidden_size=4096,
        ),
    ]

    eps = 1e-6

    all_results: list[BenchmarkResult] = []

    for case in benchmark_cases:
        case_results = run_benchmark_case(
            case=case,
            warmup_iterations=arguments.warmup,
            measurement_rounds=arguments.rounds,
            repeats_per_round=arguments.repeats,
            eps=eps,
        )

        all_results.extend(
            case_results
        )

    print_summary_table(
        all_results
    )

    output_path = save_results_to_csv(
        all_results
    )

    print(
        "\nBenchmark results saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()