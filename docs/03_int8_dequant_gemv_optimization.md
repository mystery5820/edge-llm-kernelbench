# INT8 Dequant-GEMV CUDA Optimization Report

> 日期：2026-07-13  
> 平台：NVIDIA Jetson Orin Nano Super 8GB  
> 当前阶段：PyTorch Reference、Naive CUDA、Warp-level CUDA、X-tile 实验、Vec4 CUDA 已完成

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
x:           [..., in_features], FP32 当前 CUDA 版本支持范围
weight_int8: [out_features, in_features], INT8
scale:       [out_features], FP32 per-output scale
bias:        [out_features], optional FP32
output:      [..., out_features], FP32
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
当前算子仍是 FP32 x * INT8 weight，再做 FP32 accumulation。
DP4A 需要两侧输入都按 INT8 打包；在 x 仍为 FP32 的前提下，不能直接使用 DP4A 替代当前乘加路径。
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
- empty rows；
- CPU input、FP16 CUDA input、non-contiguous input 等非法输入。

最近验证结果：

```text
MAX_JOBS=2 PYTHONPATH=python python -m pytest tests/test_int8_dequant_gemv_cuda.py -v
11 passed in 3.52s

MAX_JOBS=2 PYTHONPATH=python python -m pytest -v
138 passed in 4.57s
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
```

---

## 5. Benchmark 结果

### 5.1 Median latency

| rows | in_features | out_features | PyTorch Reference ms | CUDA Naive ms | CUDA Warp ms | CUDA Tiled ms | CUDA Vec4 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 0.587317 | 0.095019 | 0.035016 | 0.035688 | 0.032856 |
| 1 | 2048 | 2048 | 1.542368 | 0.210086 | 0.079296 | 0.102501 | 0.080854 |
| 4 | 2048 | 2048 | 1.455528 | 0.728339 | 0.228058 | 0.313299 | 0.157418 |

### 5.2 Speedup

| rows | in_features | out_features | Naive vs Reference | Warp vs Reference | Tiled vs Reference | Vec4 vs Reference | Vec4 vs Warp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 6.181x | 16.773x | 16.457x | 17.875x | 1.066x |
| 1 | 2048 | 2048 | 7.342x | 19.451x | 15.047x | 19.076x | 0.981x |
| 4 | 2048 | 2048 | 1.998x | 6.382x | 4.646x | 9.246x | 1.449x |

### 5.3 Correctness maximum error

| rows | in_features | out_features | Naive max error | Warp max error | Tiled max error | Vec4 max error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 9.91821289e-05 | 1.22070312e-04 | 1.22070312e-04 | 1.22070312e-04 |
| 1 | 2048 | 2048 | 1.22070312e-04 | 1.22070312e-04 | 1.22070312e-04 | 1.22070312e-04 |
| 4 | 2048 | 2048 | 2.44140625e-04 | 2.15530396e-04 | 2.15530396e-04 | 2.44140625e-04 |

误差来自 FP32 reduction 顺序差异，当前测试和 benchmark 使用 `1e-4` 级别容差。

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
rows=1, in=1024, out=1024：Tiled vs Warp 0.981x
rows=1, in=2048, out=2048：Tiled vs Warp 0.774x
rows=4, in=2048, out=2048：Tiled vs Warp 0.728x
```

当前判断是：每个 tile 的加载和 `__syncthreads()` 开销，超过了 8 个 warp 复用同一段 `x` 带来的收益。这个实验保留为负结果，有助于说明 shared memory 并不是这个映射下的直接收益点。

### 6.5 Vec4 优化结果

Vec4 版本保持 warp-level 的 block/warp 映射，只把 inner loop 改为 4 元向量读取。结果更接近实际瓶颈：

```text
rows=1, in=1024, out=1024：Vec4 vs Warp 1.066x
rows=1, in=2048, out=2048：Vec4 vs Warp 0.981x
rows=4, in=2048, out=2048：Vec4 vs Warp 1.449x
```

结论：

- rows=1 的场景主要受 launch、调度和全局访存影响，vec4 只能带来小幅收益或基本持平；
- rows=4 时同一 kernel 中有效工作更多，向量化读取的收益更容易体现；
- 对当前 FP32 activation 版本，`float4 + char4` 是比 DP4A 更直接的优化方向；
- 如果要真正使用 DP4A，需要增加 INT8 activation/quantization 路径，或者在上游保留 INT8 激活。

---

## 7. 当前瓶颈

当前 INT8 Dequant-GEMV 仍有明显优化空间：

1. 多个输出通道重复读取同一段 `x`；
2. 已尝试对 `x` 做 shared memory tile，但当前实现中同步开销超过复用收益；
3. Vec4 已完成，但 rows=1 场景收益有限；
4. 没有使用 DP4A，因为当前 activation 是 FP32；
5. 当前输入 `x` 仅支持 FP32；
6. 没有 FP16 / half2 路径；
7. benchmark 参数还不是与前两个算子完全一致的正式长跑参数。

---

## 8. 后续优化方向

优先级较高：

1. 增加 FP16 / half2 activation 路径；
2. 设计 INT8 activation 路径，再评估 DP4A；
3. 如果继续 x tile 复用，需要减少同步或让一个 block 计算更多 output channel；
4. 一个 block 计算更多 output channel；
5. 针对 rows=1 的 GEMV 场景做专门 kernel；
6. 重新设计 benchmark case，补充锁频、温度和 Nsight profiling 数据。

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
Vec4 版本在 rows=4, in=2048, out=2048 场景相对 Warp 达到 1.449x，在 rows=1 场景接近持平。
下一阶段最有价值的方向是 FP16/half2 activation、INT8 activation + DP4A，以及 Nsight profiling。
```
