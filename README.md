# EdgeLLM-KernelBench

面向端侧大模型推理的 CUDA 算子优化实践项目。

本项目基于 NVIDIA Jetson Orin Nano Super 8GB，围绕 LLM 推理中常见的小 batch / memory-bound GPU 算子，完成 PyTorch reference、CUDA extension、正确性测试、benchmark、Nsight profiling 和优化报告闭环。

## Final Status

当前版本已完成阶段性封版：

| Operator | Reference | CUDA baseline | Optimized variants | Tests | Benchmark | Report |
|---|---:|---:|---:|---:|---:|---:|
| RMSNorm | Done | Naive | Warp / Float4 | Done | Done | Done |
| RoPE | Done | Naive | Float4 | Done | Done | Done |
| INT8 Dequant-GEMV | Done | Naive | Warp / Tiled / Vec4 / FP16 / Half2 / Wide | Done | Done | Done |

最近验证：

```text
143 passed in 4.61s
```

## Highlights

- 实现 3 类 LLM 推理算子：RMSNorm、RoPE、INT8 Dequant-GEMV。
- 使用 PyTorch CUDA Extension 接入 C++ / CUDA kernel，并保留 Python reference 作为 correctness oracle。
- 对 RMSNorm / RoPE 实现 fused CUDA kernel 与 float4 向量化读取。
- 对 INT8 Dequant-GEMV 实现 fused dequant + GEMV，避免显式生成完整 FP32 dequant weight。
- 对 INT8 Dequant-GEMV 逐步实现 warp-level mapping、shared-memory x-tile 实验、float4/char4、FP16 activation、half2/char2、wide block 实验。
- 保留负结果分析：X-tile 和 Wide 并未形成稳定收益，文档中解释同步、调度和 occupancy tradeoff。
- 使用 Nsight Systems 量化 INT8 Dequant-GEMV 瓶颈，明确 rows=1 小 shape 中 launch / scheduling overhead 占比较高。

## Key Results

### RMSNorm

Float4 benchmark：

| rows | hidden | Float4 vs Reference | Float4 vs Naive | Float4 vs Warp |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 7.785x | 1.019x | 0.992x |
| 1 | 4096 | 7.877x | 1.016x | 0.998x |
| 16 | 4096 | 7.888x | 1.027x | 0.999x |
| 128 | 4096 | 4.588x | 1.061x | 1.049x |

### RoPE

Float4 benchmark：

| seq_len | heads | head_dim | Float4 vs Reference | Float4 vs Naive |
|---:|---:|---:|---:|---:|
| 128 | 8 | 64 | 22.404x | 1.033x |
| 512 | 8 | 64 | 15.702x | 1.006x |
| 1024 | 16 | 64 | 7.593x | 1.006x |
| 2048 | 16 | 128 | 7.317x | 1.004x |

### INT8 Dequant-GEMV

Vec4 FP32 benchmark：

| rows | in_features | out_features | Vec4 vs Reference | Vec4 vs Naive | Vec4 vs Warp |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 22.169x | 2.797x | 1.063x |
| 1 | 2048 | 2048 | 19.108x | 2.599x | 0.984x |
| 4 | 2048 | 2048 | 9.206x | 4.604x | 1.430x |

FP16 half2 benchmark：

| rows | in_features | out_features | Half2 vs Reference | Half2 vs Naive | Half2 vs Warp |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 20.229x | 2.847x | 1.004x |
| 1 | 2048 | 2048 | 17.418x | 4.555x | 0.913x |
| 4 | 2048 | 2048 | 6.885x | 7.073x | 1.076x |

Nsight Systems bottleneck summary：

| implementation | dtype | rows | in | out | kernel avg us | launch median us |
|---|---|---:|---:|---:|---:|---:|
| warp | FP32 | 1 | 1024 | 1024 | 57.113 | 41.088 |
| wide | FP32 | 1 | 1024 | 1024 | 58.056 | 41.440 |
| warp | FP32 | 1 | 2048 | 2048 | 237.469 | 46.304 |
| vec4 | FP32 | 1 | 2048 | 2048 | 225.142 | 44.544 |
| warp | FP32 | 4 | 2048 | 2048 | 590.967 | 46.368 |
| vec4 | FP32 | 4 | 2048 | 2048 | 362.124 | 40.704 |
| vec4 | FP16 | 4 | 2048 | 2048 | 594.831 | 34.944 |
| half2 | FP16 | 4 | 2048 | 2048 | 414.319 | 43.184 |

结论：rows=1 小 shape 已明显受 launch / scheduling overhead 影响；Vec4 和 Half2 的收益主要在 rows=4 时变成稳定的 kernel 级收益；Wide 不是稳定主线。

## Repository Layout

```text
python/edge_kernelbench/        Python APIs and PyTorch references
kernels/                       C++ / CUDA extension sources
tests/                         correctness tests
benchmarks/                    benchmark scripts
scripts/                       focused profiling drivers
docs/                          design notes and optimization reports
results/                       benchmark and profiling outputs
```

## Reports

- [RMSNorm Optimization](docs/01_rmsnorm_optimization.md)
- [RoPE Optimization](docs/02_rope_optimization.md)
- [INT8 Dequant-GEMV Optimization](docs/03_int8_dequant_gemv_optimization.md)
- [INT8 Dequant-GEMV Nsight Profiling](docs/04_int8_dequant_gemv_nsight_profile.md)
- [Project Benchmark Report](docs/05_benchmark_report.md)
- [Project Closure and Resume Notes](docs/06_project_closure.md)

## Environment

- Hardware: NVIDIA Jetson Orin Nano Super 8GB
- JetPack: 6.2.1
- L4T: R36.4.7
- Ubuntu: 22.04 LTS
- CUDA Toolkit: 12.6
- GPU Compute Capability: 8.7, sm_87
- Python: 3.10.12
- PyTorch: 2.8.0
- GCC/G++: 11.4.0
- CMake: 3.22.1
- Ninja: 1.10.1
- Nsight Systems: 2024.5.4
- Nsight Compute: 2024.3.1

## Run

Set `PYTHONPATH` before running tests or benchmarks:

```bash
export PYTHONPATH=python
export MAX_JOBS=2
```

Run tests:

```bash
/home/liujiayu/venvs/edge-llm-kernelbench/bin/python -m pytest -v
```

Run benchmarks:

```bash
/home/liujiayu/venvs/edge-llm-kernelbench/bin/python benchmarks/benchmark_rmsnorm.py
/home/liujiayu/venvs/edge-llm-kernelbench/bin/python benchmarks/benchmark_rope.py
/home/liujiayu/venvs/edge-llm-kernelbench/bin/python benchmarks/benchmark_int8_dequant_gemv.py
```

Run a focused Nsight Systems profile:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=results/nsight/nsys_vec4_fp32_rows1_2048 \
  /home/liujiayu/venvs/edge-llm-kernelbench/bin/python \
    scripts/profile_int8_dequant_gemv.py \
    --implementation vec4 \
    --dtype fp32 \
    --rows 1 \
    --in-features 2048 \
    --out-features 2048 \
    --warmup 20 \
    --iterations 100
```

## Resume Bullets

- Built an edge LLM CUDA kernel benchmark on Jetson Orin Nano, covering RMSNorm, RoPE, and INT8 Dequant-GEMV with PyTorch references, CUDA extensions, correctness tests, benchmark automation, and optimization reports.
- Optimized fused CUDA kernels using warp-level reduction, vectorized global memory loads, fused dequantization, and FP16 half2 loading; achieved up to 22.4x over PyTorch reference for RoPE and 22.2x for INT8 Dequant-GEMV on tested shapes.
- Profiled INT8 Dequant-GEMV with Nsight Systems and quantified launch/scheduling overhead versus kernel time, guiding the decision to stop pursuing wider block mappings and focus future work on rows=1 specialization or operator fusion.

## Closure

This repository is considered complete for the current CUDA kernel optimization phase. Valuable follow-up work would be a separate phase:

- INT8 activation + DP4A path;
- rows=1 specialized GEMV with cross-output-channel x reuse;
- Nsight Compute hardware counter collection in an environment with sufficient profiling permissions;
- TileLang / MXMACA migration study.
