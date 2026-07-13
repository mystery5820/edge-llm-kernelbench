# INT8 Dequant-GEMV Nsight Profiling Report

> 日期：2026-07-13  
> 平台：NVIDIA Jetson Orin Nano Super 8GB  
> 工具：Nsight Systems 2024.5.4, Nsight Compute 2024.3.1  
> 目标：用 Nsight 把 INT8 Dequant-GEMV 当前瓶颈量化

---

## 1. Profiling Target

Benchmark 脚本会同时包含 extension load、PyTorch reference、多个 kernel 对照和 CSV 统计，不适合作为 Nsight 直接目标。本轮新增独立 profiling driver：

```text
scripts/profile_int8_dequant_gemv.py
```

它只做三件事：

1. 构造固定 shape 的 CUDA tensor；
2. 选择一个实现反复 launch；
3. 输出 CUDA event 平均耗时和 checksum。

示例：

```bash
MAX_JOBS=2 PYTHONPATH=python nsys profile \
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

Nsight Systems summary 导出命令：

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  results/nsight/nsys_vec4_fp32_rows1_2048.nsys-rep \
  > results/nsight/nsys_vec4_fp32_rows1_2048_cuda_gpu_kern_sum.csv

nsys stats --report cuda_api_sum --format csv \
  results/nsight/nsys_vec4_fp32_rows1_2048.nsys-rep \
  > results/nsight/nsys_vec4_fp32_rows1_2048_cuda_api_sum.csv
```

---

## 2. Nsight Compute Limitation

当前用户态环境无法采 Nsight Compute hardware counter：

```text
==WARNING== Insufficient privileges to launch app for profiling. Launch app with root privileges
ncu_exit_code=0
```

记录文件：

```text
results/nsight/ncu_vec4_fp32_rows1_2048_attempt.log
```

因此本轮不能给出 SM occupancy、memory throughput、warp stall reason 等 NCU 指标。当前瓶颈量化基于 Nsight Systems 的 CUDA kernel timeline 和 CUDA API summary。

---

## 3. Nsight Systems Results

参数：

```text
warmup=20
iterations=100
```

说明：

- `event_mean_us` 来自 profiling driver 的 CUDA event；
- `nsys_kernel_avg_us` 和 `nsys_kernel_med_us` 来自 `cuda_gpu_kern_sum` 的自定义 kernel 行，包含 warmup 和 measured iterations；
- `launch_med_us` 来自 `cuda_api_sum` 的 `cudaLaunchKernel` median；
- `cudaLaunchKernel` 的平均值会被 PyTorch 初始化和首次调用放大，判断 launch 稳态成本时优先看 median。

| implementation | dtype | rows | in | out | event_mean_us | nsys_kernel_avg_us | nsys_kernel_med_us | launch_med_us |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| warp | FP32 | 1 | 1024 | 1024 | 123.004 | 57.113 | 56.960 | 41.088 |
| wide | FP32 | 1 | 1024 | 1024 | 125.884 | 58.056 | 57.888 | 41.440 |
| warp | FP32 | 1 | 2048 | 2048 | 246.322 | 237.469 | 241.104 | 46.304 |
| vec4 | FP32 | 1 | 2048 | 2048 | 242.667 | 225.142 | 241.104 | 44.544 |
| warp | FP32 | 4 | 2048 | 2048 | 570.708 | 590.967 | 712.176 | 46.368 |
| vec4 | FP32 | 4 | 2048 | 2048 | 343.840 | 362.124 | 468.880 | 40.704 |
| vec4 | FP16 | 4 | 2048 | 2048 | 575.322 | 594.831 | 716.080 | 34.944 |
| half2 | FP16 | 4 | 2048 | 2048 | 389.008 | 414.319 | 336.352 | 43.184 |

CSV 文件：

```text
results/nsight/nsys_warp_fp32_rows1_1024_cuda_gpu_kern_sum.csv
results/nsight/nsys_wide_fp32_rows1_1024_cuda_gpu_kern_sum.csv
results/nsight/nsys_warp_fp32_rows1_2048_cuda_gpu_kern_sum.csv
results/nsight/nsys_vec4_fp32_rows1_2048_cuda_gpu_kern_sum.csv
results/nsight/nsys_warp_fp32_rows4_2048_cuda_gpu_kern_sum.csv
results/nsight/nsys_vec4_fp32_rows4_2048_cuda_gpu_kern_sum.csv
results/nsight/nsys_vec4_fp16_rows4_2048_cuda_gpu_kern_sum.csv
results/nsight/nsys_half2_fp16_rows4_2048_cuda_gpu_kern_sum.csv
```

---

## 4. Bottleneck Quantification

### 4.1 rows=1, 1024x1024

```text
warp kernel median: 56.960 us
wide kernel median: 57.888 us
launch median:      ~41 us
```

这个 case 的 kernel 本体已经很短，单次 launch median 已接近 kernel 时间的 72%。因此继续减少少量指令或增加 warp/block 很难稳定转化成端到端收益。

本轮 Nsight Systems 下：

```text
wide vs warp kernel avg = 0.984x
wide vs warp event mean = 0.977x
```

这次没有复现 benchmark 中 `wide` 在 FP32 rows=1, 1024x1024 的小幅正收益，说明该优化处于噪声和调度 tradeoff 边界。

### 4.2 rows=1, 2048x2048

```text
warp kernel avg: 237.469 us
vec4 kernel avg: 225.142 us
launch median:   ~45 us
```

`vec4` 对 kernel 本体只有约 5.5% 改善：

```text
237.469 / 225.142 = 1.055x
```

这解释了 benchmark 中 rows=1 下 `vec4` 与 `warp` 基本持平：kernel 内部优化有收益，但 launch、调度、全局访存和重复读取 `x` 把收益压低。

### 4.3 rows=4, 2048x2048

FP32：

```text
warp kernel avg: 590.967 us
vec4 kernel avg: 362.124 us
kernel speedup:  1.632x
event speedup:   1.660x
```

rows=4 时单次 kernel 有更多有效工作，launch median 仍约 41-46 us，但占比明显下降。此时 `float4 + char4` 向量化读取能稳定体现出来。

FP16：

```text
vec4 FP16 fallback kernel avg: 594.831 us
half2 kernel avg:             414.319 us
kernel speedup:               1.436x
event speedup:                1.479x
```

`half2` 主要收益来自 FP16 activation 和 INT8 weight 的向量化读取；乘加仍转 FP32，以保持数值契约。Nsight Systems 结果说明 FP16 scalar fallback 是明确瓶颈，half2 路径值得保留。

---

## 5. Current Optimization Guidance

基于 Nsight Systems，本阶段判断如下：

1. `rows=1` 小 shape 的主要问题不是单条 inner loop 指令，而是 launch / 调度开销占比高，以及不同 output channel 重复读取同一段 `x`；
2. 单纯增加 warp/block 的 `wide` 方案不稳定，不能作为主线继续深挖；
3. `vec4` 对 FP32 rows=4 有明确 kernel 级收益，是当前 FP32 最有效路径；
4. `half2` 对 FP16 rows=4 有明确 kernel 级收益，是当前 FP16 最有效路径；
5. X-tile shared memory 负结果与 Nsight 结论一致：当前映射下同步开销容易吞掉 `x` 复用收益；
6. 下一步如果继续优化，应该围绕 rows=1 专门 kernel、跨 output channel 的 `x` 复用、减少 launch 次数或与上下游融合，而不是继续微调单个 warp 的读取宽度。

---

## 6. Reproducibility Notes

语法检查：

```bash
PYTHONPATH=python /home/liujiayu/venvs/edge-llm-kernelbench/bin/python \
  -m py_compile scripts/profile_int8_dequant_gemv.py
```

单 case smoke test：

```bash
MAX_JOBS=2 PYTHONPATH=python /home/liujiayu/venvs/edge-llm-kernelbench/bin/python \
  scripts/profile_int8_dequant_gemv.py \
  --implementation vec4 \
  --dtype fp32 \
  --rows 1 \
  --in-features 2048 \
  --out-features 2048 \
  --warmup 3 \
  --iterations 5
```

Nsight 原始 `.nsys-rep` 和 `.sqlite` 是可再生文件，默认由 `results/nsight/.gitignore` 忽略；轻量 CSV summary 和 NCU 权限日志保留在仓库中。
