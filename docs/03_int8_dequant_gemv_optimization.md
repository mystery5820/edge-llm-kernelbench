# INT8 Dequant-GEMV CUDA Optimization Report

> 日期：2026-07-13  
> 平台：NVIDIA Jetson Orin Nano Super 8GB  
> 当前阶段：PyTorch Reference、Naive CUDA、Warp-level CUDA 已完成

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
- empty rows；
- CPU input、FP16 CUDA input、non-contiguous input 等非法输入。

最近验证结果：

```text
MAX_JOBS=2 PYTHONPATH=python python -m pytest tests/test_int8_dequant_gemv_cuda.py -v
10 passed in 3.56s

MAX_JOBS=2 PYTHONPATH=python python -m pytest -v
137 passed in 4.57s
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
```

---

## 5. Benchmark 结果

### 5.1 Median latency

| rows | in_features | out_features | PyTorch Reference ms | CUDA Naive ms | CUDA Warp ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 0.408277 | 0.094942 | 0.037843 |
| 1 | 2048 | 2048 | 1.539941 | 0.209438 | 0.079050 |
| 4 | 2048 | 2048 | 1.453798 | 0.728413 | 0.227621 |

### 5.2 Speedup

| rows | in_features | out_features | Naive vs Reference | Warp vs Reference | Warp vs Naive |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 4.300x | 10.789x | 2.509x |
| 1 | 2048 | 2048 | 7.353x | 19.481x | 2.649x |
| 4 | 2048 | 2048 | 1.996x | 6.387x | 3.200x |

### 5.3 Correctness maximum error

| rows | in_features | out_features | Naive max error | Warp max error |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 9.91821289e-05 | 1.22070312e-04 |
| 1 | 2048 | 2048 | 1.22070312e-04 | 1.22070312e-04 |
| 4 | 2048 | 2048 | 2.44140625e-04 | 2.15530396e-04 |

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

---

## 7. 当前瓶颈

当前 INT8 Dequant-GEMV 仍有明显优化空间：

1. 多个输出通道重复读取同一段 `x`；
2. 没有对 `x` 做 shared memory tile；
3. 没有使用 `char4` / `int4` 等向量化读取 INT8 权重；
4. 没有使用 DP4A；
5. 当前输入 `x` 仅支持 FP32；
6. 没有 FP16 / half2 路径；
7. benchmark 参数还不是与前两个算子完全一致的正式长跑参数。

---

## 8. 后续优化方向

优先级较高：

1. 使用 shared memory 缓存 x tile，多个 output channel 复用；
2. INT8 权重向量化读取；
3. 尝试 DP4A 或 4 元 INT8 打包累加；
4. 一个 block 计算更多 output channel；
5. 针对 rows=1 的 GEMV 场景做专门 kernel；
6. 支持 FP16 x；
7. 重新设计 benchmark case，补充正式长跑 CSV。

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
下一阶段最有价值的方向是 x tile 复用、INT8 向量化读取和 DP4A。
```
