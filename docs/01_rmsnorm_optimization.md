# RMSNorm CUDA Optimization Report

> 日期：2026-07-12  
> 平台：NVIDIA Jetson Orin Nano Super 8GB  
> 当前阶段：PyTorch Reference、Naive CUDA、Warp Shuffle CUDA、Float4 CUDA 已完成

---

## 1. 算子背景

RMSNorm 是大语言模型中常见的归一化算子。与 LayerNorm 不同，RMSNorm 不计算均值，只基于输入向量的均方根进行缩放。

在 Transformer block 中，RMSNorm 通常出现在 Attention 或 MLP 前后：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

由于每层都会调用，RMSNorm 虽然单次计算量不大，但在端侧推理中会受到 kernel launch overhead、global memory 访问和小 batch 并行度不足的影响。

---

## 2. 数学定义

对最后一个维度上的向量：

```text
x = [x_0, x_1, ..., x_(hidden_size - 1)]
```

RMSNorm 计算过程为：

```text
mean_square = sum(x_i^2) / hidden_size
inverse_rms = 1 / sqrt(mean_square + eps)
output_i = x_i * inverse_rms * weight_i
```

当前 CUDA 实现范围：

- dtype：FP32；
- 输入：任意前置维度，最后一维为 `hidden_size`；
- 权重：一维 `[hidden_size]`；
- 输出：shape、dtype、device 与输入一致；
- 输入和权重必须是 CUDA contiguous tensor。

---

## 3. 已实现版本

### 3.1 PyTorch Reference

文件：

```text
python/edge_kernelbench/rmsnorm.py
```

作用：

- 作为数学正确性的基准；
- 支持 CPU/CUDA；
- FP16/BF16 内部使用 FP32 计算；
- 支持 autograd；
- 提供 `RMSNormReference` 模块封装。

### 3.2 Naive CUDA

文件：

```text
kernels/rmsnorm/rmsnorm_kernel.cu
```

实现策略：

- 一个 CUDA block 处理一行；
- 每个 block 固定 256 线程；
- 每个线程以 `blockDim.x` 为步长累加局部平方和；
- 使用 shared memory 二叉树规约；
- 第二次遍历写出 `x * inverse_rms * weight`。

特点：

- 结构清晰，适合作为 CUDA baseline；
- shared memory 使用量为 `256 * sizeof(float)`；
- 规约阶段包含多轮 `__syncthreads()`。

### 3.3 Warp Shuffle CUDA

文件：

```text
kernels/rmsnorm/rmsnorm_warp_kernel.cu
```

实现策略：

- 一个 CUDA block 处理一行；
- 每个 block 256 线程，即 8 个 warp；
- 第一级在 warp 内使用 `__shfl_down_sync()` 规约；
- 每个 warp 的 lane 0 写入 shared memory；
- 第一个 warp 汇总所有 warp sum；
- shared memory 使用量降低到 `8 * sizeof(float)`。

Benchmark 结论：

```text
Warp Shuffle 数值正确，但没有形成稳定加速。
```

该结论来自 `results/rmsnorm_warp_comparison_20260712_175730.csv`，必须保留，不能为了优化叙事改写。

### 3.4 Float4 CUDA

文件：

```text
kernels/rmsnorm/rmsnorm_float4_kernel.cu
```

实现策略：

- 一个 CUDA block 处理一行；
- 每个 block 固定 256 线程；
- 规约仍使用 Warp Shuffle 两级规约；
- 输入、权重和输出行首均 16 字节对齐时，使用 `float4` 访问；
- `hidden_size % 4` 的尾部 0 到 3 个元素使用标量路径；
- 当前行不满足 16 字节对齐时，整行自动回退到标量访问路径。

对齐回退是必要的：

- contiguous tensor 仍可能因为 `storage_offset != 0` 导致首地址不对齐；
- 当 `hidden_size % 4 != 0` 时，后续行首地址可能不是 16 字节对齐；
- `float4` 对未对齐地址执行向量化访问存在非法或低效风险。

---

## 4. 正确性验证

当前测试文件：

```text
tests/test_rmsnorm.py
tests/test_rmsnorm_cuda.py
tests/test_rmsnorm_warp.py
tests/test_rmsnorm_float4.py
```

验证范围：

- 固定值；
- 一维、二维、三维输入；
- 常见 hidden size：128、257、1024、4096；
- float4 尾部：1025、1026、1027；
- 全零输入；
- 常量输入；
- 空行输入；
- 输出 metadata；
- 非连续输入拒绝；
- 非对齐但 contiguous 的输入回退；
- Float4 vs Reference / Naive / Warp。

最近验证结果：

```text
MAX_JOBS=2 PYTHONPATH=python python -m pytest tests/test_rmsnorm_float4.py -v
15 passed in 3.31s

MAX_JOBS=2 PYTHONPATH=python python -m pytest -v
64 passed in 3.46s
```

---

## 5. Benchmark 方法

Benchmark 文件：

```text
benchmarks/benchmark_rmsnorm.py
```

测量方式：

- 使用 CUDA Event；
- 每轮内部重复执行多次算子，降低 Event 记录开销影响；
- 统计 mean / median / min / P95；
- speedup 使用 median 计算；
- benchmark 前先做正确性检查。

正式参数：

```text
warmup=20
rounds=30
repeats=50
```

结果文件：

```text
results/rmsnorm_baseline_20260712_161636.csv
results/rmsnorm_warp_comparison_20260712_175730.csv
results/rmsnorm_float4_comparison_20260712_221121.csv
```

说明：

本次 Float4 benchmark 使用了正式参数，但当前记录中没有执行交互式 sudo 的 `nvpmodel` / `jetson_clocks` 锁频命令。因此结果可作为 Phase 3 对比数据，但后续写正式报告或对外展示时，建议补充锁频状态截图或命令输出。

---

## 6. Phase 3 Float4 Benchmark 结果

### 6.1 Median latency

| rows | hidden | PyTorch Reference ms | Naive ms | Warp ms | Float4 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 0.236950 | 0.031014 | 0.030207 | 0.030438 |
| 1 | 4096 | 0.236925 | 0.030562 | 0.030015 | 0.030077 |
| 16 | 4096 | 0.239348 | 0.031154 | 0.030316 | 0.030342 |
| 128 | 4096 | 0.241442 | 0.055807 | 0.055214 | 0.052620 |

### 6.2 Speedup

| rows | hidden | Float4 vs Reference | Float4 vs Naive | Float4 vs Warp |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 7.785x | 1.019x | 0.992x |
| 1 | 4096 | 7.877x | 1.016x | 0.998x |
| 16 | 4096 | 7.888x | 1.027x | 0.999x |
| 128 | 4096 | 4.588x | 1.061x | 1.049x |

### 6.3 Correctness maximum error

| rows | hidden | Naive max error | Warp max error | Float4 max error |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 |
| 1 | 4096 | 4.76837158e-07 | 0.00000000e+00 | 0.00000000e+00 |
| 16 | 4096 | 1.43051147e-06 | 9.53674316e-07 | 1.43051147e-06 |
| 128 | 4096 | 1.90734863e-06 | 1.90734863e-06 | 1.90734863e-06 |

---

## 7. 性能分析

### 7.1 小 rows 场景

当 `rows=1` 或 `rows=16` 时，Float4 没有明显超过 Warp：

- `rows=1, hidden=1024`：Float4 vs Warp 为 `0.992x`；
- `rows=1, hidden=4096`：Float4 vs Warp 为 `0.998x`；
- `rows=16, hidden=4096`：Float4 vs Warp 为 `0.999x`。

主要原因：

- RMSNorm 是小算子，kernel launch overhead 占比高；
- 一个 block 只处理一行，小 rows 下 block 数量少，GPU 并行度不足；
- float4 降低了访存指令数量，但没有改变 block 数量和 launch overhead；
- 规约和同步仍然存在，访存优化收益被固定开销稀释。

### 7.2 大 rows 场景

当 `rows=128, hidden=4096` 时，Float4 有明确收益：

```text
Float4 vs Naive：1.061x
Float4 vs Warp： 1.049x
```

主要原因：

- block 数量增加，GPU 有更多并行工作；
- 每行 hidden size 较大，global memory 访问占比上升；
- float4 向量化能减少访存指令数量，提高连续访问效率；
- 固定开销被更多行摊薄。

### 7.3 Warp 版本结论保留

Warp Shuffle 降低了 shared memory 使用和 block 级同步次数，但在当前 Jetson Orin Nano 测量中没有形成稳定加速。

这说明对小算子来说，理论上减少规约开销并不一定能直接转化为端到端 latency 优势。实际性能还受到：

- launch overhead；
- occupancy；
- memory latency；
- block 数量；
- 编译器优化；
- PyTorch extension 调用路径；
- Jetson 当前频率状态。

---

## 8. 当前瓶颈

当前 RMSNorm FP32 实现的主要瓶颈：

1. 小 rows 下并行度不足；
2. 每一行一个 block，rows 少时 SM 利用率有限；
3. 每次 RMSNorm 仍需独立 kernel launch；
4. FP32 路径不是 LLM 推理最常见的最终形态；
5. 当前未实现 half / half2；
6. 还没有融合 residual、scale、activation 等上下游操作；
7. 没有针对 hidden_size 小于或等于 1024 的专门策略；
8. benchmark 尚未系统记录锁频状态、功耗模式和多次运行波动。

---

## 9. 后续优化方向

优先级较高的方向：

1. FP16 输入、FP32 累加、FP16 输出；
2. `half2` 向量化；
3. 一个 warp 处理一行，适配小 hidden size 和小 rows 场景；
4. 一个 block 处理多行，提高 rows 小时的 SM 利用率；
5. 针对 `hidden_size=1024/4096` 做专门模板化实现；
6. 融合 residual add：

```text
output = residual + rmsnorm(x, weight)
```

7. 融合后续 elementwise 操作，减少 launch 和中间 tensor 写回；
8. 记录 `nvpmodel`、`jetson_clocks`、温度和频率，形成更完整的 benchmark metadata。

建议下一步：

```text
短线：补一次锁频状态下的 RMSNorm benchmark，确认 Float4 结果稳定性。
中线：进入 RoPE CUDA 算子，完成第二个算子的 reference / naive / benchmark 闭环。
长线：回到 RMSNorm 做 FP16/half2 和融合版本。
```

---

## 10. 当前阶段结论

RMSNorm 已完成从 PyTorch Reference 到三种 CUDA 实现的闭环：

```text
PyTorch Reference
    ↓
Naive CUDA
    ↓
Warp Shuffle CUDA
    ↓
Float4 Vectorized CUDA
```

最终结论：

```text
Naive CUDA 相比 PyTorch Reference 有稳定加速。
Warp Shuffle 数值正确，但没有形成稳定加速。
Float4 数值正确，在 rows=128, hidden=4096 场景有明确收益；
在小 rows 场景与 Warp 基本持平。
```

RMSNorm 当前已经适合作为项目第一个完整算子案例保留。后续可以基于该结构继续扩展 RoPE 和 INT8 Dequant-GEMV。
