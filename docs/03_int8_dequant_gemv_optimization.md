# INT8 Dequant-GEMV CUDA Optimization Report

> 日期：2026-07-13  
> 平台：NVIDIA Jetson Orin Nano Super 8GB  
> 当前阶段：PyTorch Reference、Naive CUDA、Warp-level CUDA、X-tile 实验、Vec4 CUDA、FP16 activation、Half2 CUDA、Wide CUDA 实验、Nsight Systems profiling 已完成

---

## 1. 算子背景

INT8 Dequant-GEMV 面向量化大模型推理中的权重反量化与矩阵向量乘。

典型计算：

```text
dequant_weight = weight_int8.float() * scale[:, None]
output = x @ dequant_weight.T + bias
```

其中：

```text
x:           [..., in_features], FP32 / FP16 当前 CUDA 版本支持范围
weight_int8: [out_features, in_features], INT8
scale:       [out_features], FP32 per-output scale
bias:        [out_features], optional FP32
output:      [..., out_features], 与 x dtype 一致
```

这个算子的重点不是单独做矩阵乘，而是避免显式生成完整 `dequant_weight` 后再调用通用 matmul。CUDA kernel 可以把反量化和 GEMV 累加融合到一个过程里。

---

## 2. 已实现版本

### 2.1 PyTorch Reference

文件：

```text
python/edge_kernelbench/int8_dequant_gemv.py
```

API：

```python
int8_dequant_gemv_reference(x, weight_int8, scale, bias=None)
INT8DequantGEMVReference(weight_int8, scale, bias=None)
```

特点：

- 支持 CPU / CUDA；
- 支持 `x` 形状 `[..., in_features]`；
- 支持 optional bias；
- FP16 / BF16 输入内部使用 FP32 计算，输出恢复输入 dtype；
- 支持对 `x`、`scale`、`bias` 反向传播。

### 2.2 Naive CUDA

文件：

```text
kernels/int8_dequant_gemv/int8_dequant_gemv_kernel.cu
```

API：

```python
int8_dequant_gemv_cuda(x, weight_int8, scale, bias=None)
```

实现策略：

```text
一个 CUDA block 计算一个 output[row, out_feature]
block 内 256 个线程遍历 in_features
shared memory 做 block 级 FP32 reduction
最后乘 scale[out_feature] 并加 bias[out_feature]
```

优点：

- 逻辑直接；
- 正确性容易验证；
- 相比 PyTorch Reference 避免显式 dequant_weight 中间张量。

限制：

- block 数为 `rows * out_features`；
- out_features 大时 block 数量很多；
- 每个输出通道独立启动一个 block，调度开销较高。

### 2.3 Warp-level CUDA

文件：

```text
kernels/int8_dequant_gemv/int8_dequant_gemv_warp_kernel.cu
```

API：

```python
int8_dequant_gemv_cuda_warp(x, weight_int8, scale, bias=None)
```

实现策略：

```text
一个 block = 8 个 warp = 256 个线程
每个 warp 计算一个 output[row, out_feature]
一个 block 同时计算 8 个输出通道
warp 内使用 __shfl_down_sync 做 FP32 reduction
```

相比 Naive：

- block 数大约减少 8 倍；
- 不需要 shared memory；
- warp 内规约减少同步开销；
- 对 GEMV 中大量 out_feature 的场景更合适。

### 2.4 X-tile CUDA experiment

文件：

```text
kernels/int8_dequant_gemv/int8_dequant_gemv_tiled_kernel.cu
```

API：

```python
int8_dequant_gemv_cuda_tiled(x, weight_int8, scale, bias=None)
```

实验目标：

```text
在 warp-level 映射基础上，将 x 的一段 tile 加载到 shared memory，
让同一个 block 内的 8 个 warp 复用这段 x。
```

实验结果：

```text
数值正确，但冒烟 benchmark 中慢于 warp-level 版本。
```

原因分析：

```text
shared memory 加载和每个 tile 的 __syncthreads() 带来的开销，
超过了 8 个 warp 复用 x tile 带来的收益。
当前 GEMV case 中，warp-level 版本保持 x 的直接 global load 反而更快。
```

### 2.5 Vec4 CUDA

文件：

```text
kernels/int8_dequant_gemv/int8_dequant_gemv_vec4_kernel.cu
```

API：

```python
int8_dequant_gemv_cuda_vec4(x, weight_int8, scale, bias=None)
```

实现策略：

```text
保持 warp-level 映射：
一个 block = 8 个 warp
每个 warp 计算一个 output[row, out_feature]

当 in_features % 4 == 0 且指针对齐时：
x 使用 float4 读取
weight_int8 使用 char4 读取
每个 lane 一次处理 4 个连续 column

否则回退到标量 warp 路径。
```

说明：

```text
当前算子仍是 FP32/FP16 activation * INT8 weight，再做 FP32 accumulation。
DP4A 需要两侧输入都按 INT8 打包；在 x 仍为 FP32/FP16 的前提下，不能直接使用 DP4A 替代当前乘加路径。
```

### 2.6 FP16 activation support

实现范围：

```text
CUDA Naive
CUDA Warp-level
CUDA X-tile
CUDA Vec4 fallback path
```

实现策略：

```text
x 支持 FP32 / FP16
weight_int8 保持 INT8
scale / bias 保持 FP32
内部 dot product 和 reduction 使用 FP32 accumulation
output dtype 与 x dtype 一致
```

说明：

```text
当前 FP16 activation 是功能支持，不是 half2 优化。
Vec4 的 float4 + char4 快路径仍只用于 FP32 x；
FP16 x 在 Vec4 API 下会走标量 warp fallback。
```

### 2.7 Half2 CUDA

文件：

```text
kernels/int8_dequant_gemv/int8_dequant_gemv_half2_kernel.cu
```

API：

```python
int8_dequant_gemv_cuda_half2(x, weight_int8, scale, bias=None)
```

实现策略：

```text
仅支持 FP16 x
保持 warp-level 映射
当 in_features % 2 == 0 且指针对齐时：
    x 使用 half2 读取
    weight_int8 使用 char2 读取
    每个 lane 一次处理 2 个连续 column
    half2 读取后拆成 FP32 乘加，保持 FP32 accumulation
否则回退到标量 half warp 路径
```

说明：

```text
最初直接用 half2 乘法会引入更大的 FP16 product 舍入误差。
当前版本只用 half2/char2 做向量化读取，乘加仍转为 FP32，
以保持与 reference 和标量 CUDA 路径一致的数值契约。
```

### 2.8 Wide CUDA experiment

文件：

```text
kernels/int8_dequant_gemv/int8_dequant_gemv_wide_kernel.cu
```

API：

```python
int8_dequant_gemv_cuda_wide(x, weight_int8, scale, bias=None)
```

实验目标：

```text
把 warp-level 映射从 8 warp/block 扩展到 16 warp/block。

warp-level：一个 block 计算 8 个 output channel
wide：      一个 block 计算 16 个 output channel
```

实验目的：

```text
验证继续减少 block 数是否能改善 rows=1 / large out_features 场景。
```

---

## 3. 正确性验证

测试文件：

```text
tests/test_int8_dequant_gemv.py
tests/test_int8_dequant_gemv_cuda.py
```

覆盖范围：

- 固定值；
- 1D / 2D / 3D `x`；
- optional bias；
- FP16 CUDA reference；
- module 封装；
- autograd；
- CUDA Naive vs Reference；
- CUDA Warp vs Reference / Naive；
- CUDA Tiled vs Reference / Warp；
- CUDA Vec4 vs Reference / Warp；
- Vec4 标量 fallback 路径；
- FP16 activation CUDA Naive / Warp / Tiled / Vec4；
- CUDA Half2 vs Reference / Warp；
- Half2 标量 fallback 路径；
- CUDA Wide vs Reference / Warp；
- empty rows；
- CPU input、float64 CUDA input、non-contiguous input 等非法输入。

最近验证结果：

```text
MAX_JOBS=2 PYTHONPATH=python python -m pytest tests/test_int8_dequant_gemv_cuda.py -v
16 passed in 3.62s

MAX_JOBS=2 PYTHONPATH=python python -m pytest -v
143 passed in 4.68s
```

---

## 4. Benchmark 方法

Benchmark 文件：

```text
benchmarks/benchmark_int8_dequant_gemv.py
```

本轮参数：

```text
warmup=5
rounds=10
repeats=10
```

说明：

与 RMSNorm / RoPE 不同，INT8 Dequant-GEMV 的 naive kernel 使用大量 block。默认 `warmup=20, rounds=30, repeats=50` 会显著拉长运行时间，因此本轮使用较轻但明确记录的参数生成阶段性结果。

结果文件：

```text
results/int8_dequant_gemv_warp_comparison_20260713_122659.csv
results/int8_dequant_gemv_warp_comparison_console_20260713_122655.log
results/int8_dequant_gemv_vec4_comparison_20260713_131217.csv
results/int8_dequant_gemv_vec4_comparison_console_20260713_131213.log
results/int8_dequant_gemv_fp32_vec4_comparison_20260713_135940.csv
results/int8_dequant_gemv_fp32_vec4_comparison_console_20260713_135935.log
results/int8_dequant_gemv_fp16_vec4_comparison_20260713_135528.csv
results/int8_dequant_gemv_fp16_vec4_comparison_console_20260713_135523.log
results/int8_dequant_gemv_fp16_half2_comparison_20260713_142951.csv
results/int8_dequant_gemv_fp16_half2_comparison_console_20260713_142946.log
results/int8_dequant_gemv_fp32_wide_comparison_20260713_144325.csv
results/int8_dequant_gemv_fp32_wide_comparison_console_20260713_144321.log
results/int8_dequant_gemv_fp16_wide_comparison_20260713_144341.csv
results/int8_dequant_gemv_fp16_wide_comparison_console_20260713_144337.log
results/nsight/*_cuda_gpu_kern_sum.csv
results/nsight/*_cuda_api_sum.csv
```

---

## 5. Benchmark 结果

### 5.1 Median latency

| rows | in_features | out_features | PyTorch Reference ms | CUDA Naive ms | CUDA Warp ms | CUDA Tiled ms | CUDA Vec4 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 0.751851 | 0.094877 | 0.036046 | 0.036555 | 0.033915 |
| 1 | 2048 | 2048 | 1.541912 | 0.209755 | 0.079397 | 0.102296 | 0.080693 |
| 4 | 2048 | 2048 | 1.457210 | 0.728774 | 0.226344 | 0.313338 | 0.158288 |

### 5.2 Speedup

| rows | in_features | out_features | Naive vs Reference | Warp vs Reference | Tiled vs Reference | Vec4 vs Reference | Vec4 vs Warp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 7.925x | 20.858x | 20.568x | 22.169x | 1.063x |
| 1 | 2048 | 2048 | 7.351x | 19.420x | 15.073x | 19.108x | 0.984x |
| 4 | 2048 | 2048 | 2.000x | 6.438x | 4.651x | 9.206x | 1.430x |

### 5.3 Correctness maximum error

| rows | in_features | out_features | Naive max error | Warp max error | Tiled max error | Vec4 max error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 9.91821289e-05 | 1.22070312e-04 | 1.22070312e-04 | 1.22070312e-04 |
| 1 | 2048 | 2048 | 1.22070312e-04 | 1.22070312e-04 | 1.22070312e-04 | 1.22070312e-04 |
| 4 | 2048 | 2048 | 2.44140625e-04 | 2.15530396e-04 | 2.15530396e-04 | 2.44140625e-04 |

误差来自 FP32 reduction 顺序差异，当前测试和 benchmark 使用 `1e-4` 级别容差。

### 5.4 FP16 activation benchmark

FP16 activation 路径使用相同 benchmark 参数：

```text
warmup=5
rounds=10
repeats=10
```

| rows | in_features | out_features | PyTorch Reference ms | CUDA Naive ms | CUDA Warp ms | CUDA Tiled ms | CUDA Vec4 API ms | CUDA Half2 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 0.719675 | 0.101286 | 0.035707 | 0.036190 | 0.035797 | 0.035576 |
| 1 | 2048 | 2048 | 1.564176 | 0.409086 | 0.081986 | 0.102133 | 0.078826 | 0.089803 |
| 4 | 2048 | 2048 | 1.475200 | 1.515448 | 0.230608 | 0.313699 | 0.228714 | 0.214272 |

| rows | in_features | out_features | Vec4 API vs Warp | Half2 vs Reference | Half2 vs Naive | Half2 vs Warp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 0.997x | 20.229x | 2.847x | 1.004x |
| 1 | 2048 | 2048 | 1.040x | 17.418x | 4.555x | 0.913x |
| 4 | 2048 | 2048 | 1.008x | 6.885x | 7.073x | 1.076x |

这里的 `CUDA Vec4 API` 对 FP16 x 会走标量 warp fallback，而不是 float4 快路径。

### 5.5 Wide experiment benchmark

FP32 wide 结果：

| rows | in_features | out_features | CUDA Warp ms | CUDA Vec4 ms | CUDA Wide ms | Wide vs Warp | Wide vs Vec4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 0.037581 | 0.035738 | 0.034914 | 1.076x | 1.024x |
| 1 | 2048 | 2048 | 0.079610 | 0.080504 | 0.080648 | 0.987x | 0.998x |
| 4 | 2048 | 2048 | 0.228034 | 0.157589 | 0.235867 | 0.967x | 0.668x |

FP16 wide 结果：

| rows | in_features | out_features | CUDA Warp ms | CUDA Vec4 API ms | CUDA Half2 ms | CUDA Wide ms | Wide vs Warp | Wide vs Best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 0.033698 | 0.032477 | 0.032632 | 0.033283 | 1.012x | 0.976x |
| 1 | 2048 | 2048 | 0.082370 | 0.078840 | 0.082358 | 0.082856 | 0.994x | 0.951x |
| 4 | 2048 | 2048 | 0.230562 | 0.228766 | 0.214141 | 0.239061 | 0.964x | 0.896x |

---

## 6. 性能分析

### 6.1 PyTorch Reference 的开销

Reference 会显式构造：

```text
dequant_weight = weight_int8.float() * scale[:, None]
```

这会产生完整 FP32 权重矩阵，再执行 matmul。对于 `2048 x 2048` 权重，这个中间张量约为：

```text
2048 * 2048 * 4 bytes = 16 MB
```

CUDA kernel 将反量化和 dot product 融合，避免写出完整 dequant 权重，因此相比 Reference 有明显收益。

### 6.2 Naive CUDA 的问题

Naive 每个输出元素一个 block：

```text
blocks = rows * out_features
```

当 `out_features=2048` 且 `rows=4` 时：

```text
blocks = 8192
```

每个 block 都有 256 线程和 shared memory reduction，调度和同步成本明显。

### 6.3 Warp-level 优化收益

Warp 版本每个 block 计算 8 个输出通道：

```text
blocks = rows * ceil(out_features / 8)
```

因此 block 数大约减少 8 倍。实际 benchmark 中：

```text
Warp vs Naive = 2.509x 到 3.200x
```

没有达到理论 8 倍的原因：

- 每个 warp 只有 32 线程参与一个输出通道的 reduction；
- global memory 读取仍然占主导；
- 不同输出通道重复读取同一行 x；
- 没有利用 shared memory 缓存 x tile；
- INT8 权重读取和 FP32 累加仍是标量路径。

### 6.4 X-tile 实验结果

X-tile 版本试图让同一个 block 内 8 个 warp 复用 shared memory 中的 `x` tile。结果数值正确，但三组 benchmark 都慢于 warp-level：

```text
rows=1, in=1024, out=1024：Tiled vs Warp 0.986x
rows=1, in=2048, out=2048：Tiled vs Warp 0.776x
rows=4, in=2048, out=2048：Tiled vs Warp 0.722x
```

当前判断是：每个 tile 的加载和 `__syncthreads()` 开销，超过了 8 个 warp 复用同一段 `x` 带来的收益。这个实验保留为负结果，有助于说明 shared memory 并不是这个映射下的直接收益点。

### 6.5 Vec4 优化结果

Vec4 版本保持 warp-level 的 block/warp 映射，只把 inner loop 改为 4 元向量读取。结果更接近实际瓶颈：

```text
rows=1, in=1024, out=1024：Vec4 vs Warp 1.063x
rows=1, in=2048, out=2048：Vec4 vs Warp 0.984x
rows=4, in=2048, out=2048：Vec4 vs Warp 1.430x
```

结论：

- rows=1 的场景主要受 launch、调度和全局访存影响，vec4 只能带来小幅收益或基本持平；
- rows=4 时同一 kernel 中有效工作更多，向量化读取的收益更容易体现；
- 对当前 FP32 activation 版本，`float4 + char4` 是比 DP4A 更直接的优化方向；
- 如果要真正使用 DP4A，需要增加 INT8 activation/quantization 路径，或者在上游保留 INT8 激活。

### 6.6 FP16 activation 结果

FP16 activation 支持的主要价值是补齐端侧推理常见 dtype，而不是立即获得 half2 性能收益。当前实现每次读取 FP16 x 后转成 FP32 参与乘加和规约，最后再写回 FP16 output。

结果上：

- Warp 和 Vec4 API 在 FP16 下基本持平，因为 Vec4 API 对 FP16 走 scalar fallback；
- Naive FP16 在 rows=4, in=2048, out=2048 场景慢于 PyTorch Reference，说明一个输出元素一个 block 的映射在半精度输入下仍不合适；
- 真正的 FP16 性能优化需要 half2 读取，或者进一步设计更适合 half2 的计算路径，而不是只把 scalar_t 换成 half。

### 6.7 Half2 结果

Half2 版本不是全面加速：

```text
rows=1, in=1024, out=1024：Half2 vs Warp 1.004x
rows=1, in=2048, out=2048：Half2 vs Warp 0.913x
rows=4, in=2048, out=2048：Half2 vs Warp 1.076x
```

结论：

- rows=1 场景仍主要受 launch、调度和全局访存影响，half2 读取不足以稳定领先；
- rows=4 时有效工作更多，half2/char2 向量化读取能体现出小幅收益；
- 为了保持 FP32 accumulation 的数值契约，当前没有使用 half2 乘法做最终 product，因此 half2 算术吞吐收益有限；
- 如果要继续优化 FP16，需要考虑一个 block 计算更多 output channel、减少重复读取，或引入更激进但需要重新定义误差边界的 FP16 product 路径。

### 6.8 Wide 实验结果

Wide 版本把每个 block 的 warp 数从 8 提高到 16，继续减少 block 数。结果不是稳定收益：

```text
FP32 rows=1, in=1024, out=1024：Wide vs Warp 1.076x
FP32 rows=1, in=2048, out=2048：Wide vs Warp 0.987x
FP32 rows=4, in=2048, out=2048：Wide vs Warp 0.967x

FP16 rows=1, in=1024, out=1024：Wide vs Warp 1.012x
FP16 rows=1, in=2048, out=2048：Wide vs Warp 0.994x
FP16 rows=4, in=2048, out=2048：Wide vs Warp 0.964x
```

结论：

- 减少 block 数在 `rows=1, in=1024, out=1024` 这种较小 case 有小幅收益；
- 当 `in_features=2048` 或 `rows=4` 时，512 threads/block 带来的 occupancy / scheduling tradeoff 抵消了 block 数减少收益；
- 对 FP32 rows=4，Vec4 仍是最佳路径；
- 对 FP16 rows=4，Half2 仍是最佳路径；
- 继续简单增加 warp/block 不是主要优化方向，后续更应转向 rows=1 专门映射、跨 output channel 的 x 复用，或 Nsight profiling。

### 6.9 Nsight Systems profiling

Profiling 报告：

```text
docs/04_int8_dequant_gemv_nsight_profile.md
```

本轮使用独立 driver 避免 benchmark 中 PyTorch reference、extension load、多实现循环干扰：

```text
scripts/profile_int8_dequant_gemv.py
```

Nsight Compute 当前受权限限制，无法采 hardware counter / occupancy：

```text
==WARNING== Insufficient privileges to launch app for profiling. Launch app with root privileges
```

Nsight Systems 已成功采集 CUDA timeline 和 kernel summary。关键量化结果：

| implementation | dtype | rows | in | out | event mean us | nsys kernel avg us | launch median us |
|---|---|---:|---:|---:|---:|---:|---:|
| warp | FP32 | 1 | 1024 | 1024 | 123.004 | 57.113 | 41.088 |
| wide | FP32 | 1 | 1024 | 1024 | 125.884 | 58.056 | 41.440 |
| warp | FP32 | 1 | 2048 | 2048 | 246.322 | 237.469 | 46.304 |
| vec4 | FP32 | 1 | 2048 | 2048 | 242.667 | 225.142 | 44.544 |
| warp | FP32 | 4 | 2048 | 2048 | 570.708 | 590.967 | 46.368 |
| vec4 | FP32 | 4 | 2048 | 2048 | 343.840 | 362.124 | 40.704 |
| vec4 | FP16 | 4 | 2048 | 2048 | 575.322 | 594.831 | 34.944 |
| half2 | FP16 | 4 | 2048 | 2048 | 389.008 | 414.319 | 43.184 |

结论：

- `rows=1, 1024x1024` 下 custom kernel median 约 57 us，`cudaLaunchKernel` median 约 41 us，launch 已接近 kernel 时间的 72%；
- `rows=1, 2048x2048` 下 Vec4 kernel avg 只比 Warp 快约 5.5%，所以 benchmark 中基本持平是合理结果；
- `rows=4, 2048x2048` 下 Vec4 FP32 kernel avg 相比 Warp 约 1.632x，说明向量化读取在有效工作量增加后才稳定体现；
- FP16 rows=4 下 Half2 相比 Vec4 fallback kernel avg 约 1.436x，说明 FP16 scalar fallback 是明确瓶颈；
- Wide 在 Nsight Systems 下没有复现 rows=1 小 case 的稳定正收益，继续增加 warp/block 不是主线。

---

## 7. 当前瓶颈

当前 INT8 Dequant-GEMV 仍有明显优化空间：

1. 多个输出通道重复读取同一段 `x`；
2. 已尝试对 `x` 做 shared memory tile，但当前实现中同步开销超过复用收益；
3. Vec4 已完成，但 rows=1 场景收益有限；
4. 没有使用 DP4A，因为当前 activation 不是 INT8 packed activation；
5. Half2 已支持，但 rows=1 场景没有稳定收益；
6. Wide 已验证，继续增加 warp/block 没有稳定收益；
7. Nsight Systems 已量化 launch / kernel 时间，但 Nsight Compute hardware counter 受权限限制暂未采到；
8. Vec4 快路径仍只支持 FP32 activation；
9. benchmark 参数还不是与前两个算子完全一致的正式长跑参数。

---

## 8. 后续优化方向

优先级较高：

1. 针对 rows=1 的 GEMV 场景做专门 kernel；
2. 设计跨 output channel 的 x 复用策略，而不是单纯增加 warp/block；
3. 设计 INT8 activation 路径，再评估 DP4A；
4. 如果继续 x tile 复用，需要减少同步或改变 block 内协作方式；
5. 需要 root 或放开 perf counter 权限后，用 Nsight Compute 补充 occupancy、memory throughput 和 warp stall reason；
6. 重新设计 benchmark case，补充锁频、温度和更长时间 profiling 数据。

---

## 9. 当前阶段结论

INT8 Dequant-GEMV 已完成第三个算子的基础闭环：

```text
PyTorch Reference
    ↓
Naive CUDA
    ↓
Warp-level CUDA
    ↓
X-tile CUDA experiment
    ↓
Vec4 CUDA
    ↓
FP16 activation support
    ↓
Half2 CUDA
    ↓
Wide CUDA experiment
    ↓
Correctness Tests
    ↓
Benchmark CSV / Console Log
    ↓
Optimization Report
```

当前结论：

```text
CUDA Naive 避免显式 dequant_weight，中等规模下相对 PyTorch Reference 有明显收益。
Warp-level 版本通过一个 block 计算 8 个输出通道，进一步获得 2.5x 到 3.2x 的 Naive 相对加速。
X-tile shared-memory 复用实验数值正确但没有加速，说明当前映射下同步开销更关键。
Vec4 版本在 rows=4, in=2048, out=2048 场景相对 Warp 达到 1.430x，在 rows=1 场景接近持平。
FP16 activation 已完成基础功能支持，但当前还不是 half2 优化。
Half2 版本在 rows=4, in=2048, out=2048 场景相对 Warp 达到 1.076x，但 rows=1 场景没有稳定收益。
Wide 版本证明继续单纯增加 warp/block 没有稳定收益。
Nsight Systems profiling 量化了 rows=1 的 launch 占比、Vec4 rows=4 的 kernel 级收益和 Half2 rows=4 的瓶颈改善。
下一阶段最有价值的方向是 rows=1 专门 kernel、跨 output channel 的 x 复用、INT8 activation + DP4A，以及 Nsight Compute hardware counter 补采。
```
