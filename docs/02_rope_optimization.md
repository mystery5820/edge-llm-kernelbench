# RoPE CUDA Optimization Report

> 日期：2026-07-13  
> 平台：NVIDIA Jetson Orin Nano Super 8GB  
> 当前阶段：PyTorch Reference、Naive CUDA、Float4 CUDA 已完成

---

## 1. 算子背景

RoPE 全称 Rotary Position Embedding，即旋转位置编码。它常用于 Transformer Attention 中的 query 和 key，通过对 head_dim 上的 even / odd 维度成对旋转，把 token 位置信息注入到注意力计算中。

典型输入：

```text
q:   [seq_len, num_heads, head_dim]
k:   [seq_len, num_heads, head_dim]
cos: [seq_len, head_dim / 2]
sin: [seq_len, head_dim / 2]
```

本项目中 RoPE 也支持：

```text
[seq_len, head_dim]
[batch_size, seq_len, num_heads, head_dim]
```

---

## 2. 数学定义

对每个 even / odd 维度对：

```text
x_even = x[..., 2i]
x_odd  = x[..., 2i + 1]
```

RoPE 旋转为：

```text
y_even = x_even * cos_i - x_odd * sin_i
y_odd  = x_even * sin_i + x_odd * cos_i
```

同一组 `cos/sin` 同时作用于 q 和 k。

---

## 3. 已实现版本

### 3.1 PyTorch Reference

文件：

```text
python/edge_kernelbench/rope.py
```

API：

```python
apply_rope_reference(x, cos, sin)
rope_reference(q, k, cos, sin)
RoPEReference()
```

特点：

- 支持 CPU / CUDA；
- 支持 2D / 3D / 4D 常见形状；
- 支持 `cos/sin` 为 `[head_dim / 2]` 或 `[seq_len, head_dim / 2]`；
- FP16 / BF16 输入内部使用 FP32 计算，输出恢复输入 dtype；
- 支持 autograd。

### 3.2 Naive CUDA

文件：

```text
kernels/rope/rope.cpp
kernels/rope/rope_kernel.cu
python/edge_kernelbench/rope_cuda.py
```

API：

```python
rope_cuda(q, k, cos, sin)
```

实现策略：

- 当前只支持 CUDA contiguous FP32；
- 每个线程处理一个 even/odd pair；
- 同一个线程同时计算 q 和 k；
- 支持 `[head_dim / 2]` 和 `[seq_len, head_dim / 2]` 两种 `cos/sin`；
- 空序列直接返回空输出。

线程映射：

```text
pair_index -> row_id + pair_in_row
row_id     -> sequence_id + head_id
```

对 `[batch, seq, heads, head_dim]`：

```text
sequence_id = (row_id / num_heads) % seq_len
```

### 3.3 Float4 CUDA

文件：

```text
kernels/rope/rope_float4_kernel.cu
```

API：

```python
rope_cuda_float4(q, k, cos, sin)
```

实现策略：

- 常见 `head_dim=64/128` 满足 `head_dim % 4 == 0`；
- 每个线程使用一个 `float4` 处理两个 even/odd pair；
- q 和 k 同时处理；
- q/k/output 基址均 16 字节对齐时走 float4；
- `head_dim % 4 != 0` 或基址不对齐时回退标量路径；
- 仍复用 C++ 公共输入检查。

---

## 4. 正确性验证

测试文件：

```text
tests/test_rope.py
tests/test_rope_cuda.py
tests/test_rope_float4.py
```

覆盖范围：

- 固定 90 度旋转；
- 2D / 3D / 4D 输入；
- q/k 同时处理；
- `cos/sin` 1D 广播；
- FP16 CUDA reference；
- autograd；
- CUDA Naive vs Reference；
- CUDA Float4 vs Reference / Naive；
- `head_dim=10` 标量回退；
- `storage_offset=1` 非对齐回退；
- 空序列；
- CPU/FP16/non-contiguous/odd head_dim/wrong sequence length 等非法输入。

最近验证结果：

```text
MAX_JOBS=2 PYTHONPATH=python python -m pytest tests/test_rope_float4.py -v
10 passed in 3.28s

MAX_JOBS=2 PYTHONPATH=python python -m pytest -v
110 passed in 3.89s
```

---

## 5. Benchmark 方法

Benchmark 文件：

```text
benchmarks/benchmark_rope.py
```

测量方式：

- CUDA Event；
- 每轮重复执行多次算子；
- 统计 mean / median / min / P95；
- speedup 使用 median 延迟计算；
- benchmark 前做正确性检查。

正式参数：

```text
warmup=20
rounds=30
repeats=50
```

结果文件：

```text
results/rope_naive_comparison_20260712_230056.csv
results/rope_float4_comparison_20260712_231133.csv
```

说明：

本次记录使用正式参数，但没有执行需要 sudo 的 `nvpmodel` / `jetson_clocks` 锁频命令。对外展示前建议补充锁频状态和温度/频率信息。

---

## 6. Benchmark 结果

### 6.1 Median latency

| seq_len | heads | head_dim | PyTorch Reference ms | CUDA Naive ms | CUDA Float4 ms |
|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 64 | 0.807030 | 0.037198 | 0.036022 |
| 512 | 8 | 64 | 0.809436 | 0.051836 | 0.051549 |
| 1024 | 16 | 64 | 1.380095 | 0.182848 | 0.181752 |
| 2048 | 16 | 128 | 5.161007 | 0.708220 | 0.705388 |

### 6.2 Speedup

| seq_len | heads | head_dim | Float4 vs Reference | Float4 vs Naive |
|---:|---:|---:|---:|---:|
| 128 | 8 | 64 | 22.404x | 1.033x |
| 512 | 8 | 64 | 15.702x | 1.006x |
| 1024 | 16 | 64 | 7.593x | 1.006x |
| 2048 | 16 | 128 | 7.317x | 1.004x |

### 6.3 Correctness maximum error

| seq_len | heads | head_dim | Naive max error | Float4 max error |
|---:|---:|---:|---:|---:|
| 128 | 8 | 64 | 4.76837158e-07 | 2.38418579e-07 |
| 512 | 8 | 64 | 4.76837158e-07 | 4.76837158e-07 |
| 1024 | 16 | 64 | 4.76837158e-07 | 4.76837158e-07 |
| 2048 | 16 | 128 | 4.76837158e-07 | 4.76837158e-07 |

---

## 7. 性能分析

### 7.1 Naive CUDA 收益明显

RoPE PyTorch Reference 使用多个 PyTorch elementwise / indexing 操作组合完成：

```text
even/odd slice
乘 cos/sin
加减
重新写回 even/odd
```

这些操作会带来多个 kernel launch 和中间 tensor 读写。CUDA Naive 将 q/k 的旋转融合到一个 kernel 中，因此相对 PyTorch Reference 有明显加速：

```text
7.289x 到 21.436x
```

### 7.2 Float4 收益稳定但幅度较小

Float4 相对 Naive 的加速为：

```text
1.004x 到 1.033x
```

收益较小的原因：

1. Naive 每个线程已经处理一个连续 even/odd pair，访存模式较简单；
2. RoPE 每个 pair 都需要读取对应 cos/sin，float4 只减少 q/k/output 的部分访存指令；
3. 计算量很轻，整体仍受访存、launch 和调度影响；
4. 对大规模输入，global memory 带宽和并行度已经较充分，float4 进一步收益有限。

### 7.3 小规模 case 收益更明显

`seq=128, heads=8, head_dim=64` 中：

```text
Float4 vs Naive = 1.033x
```

这个 case 的数据量较小，float4 减少指令数量后更容易反映到 median latency 上。随着输入规模增大，收益收敛到约 1% 或更低。

---

## 8. 当前瓶颈

当前 RoPE 实现仍有以下限制：

1. CUDA kernel 只支持 FP32；
2. LLM 推理中更常见的是 FP16/BF16；
3. `cos/sin` 仍按标量读取；
4. q/k 单独输入输出，没有与 attention 前后操作融合；
5. 没有 half2；
6. 没有把 q/k layout 与实际模型框架做进一步适配；
7. benchmark metadata 尚未记录锁频、温度、功耗模式。

---

## 9. 后续优化方向

优先级较高：

1. FP16 输入输出；
2. `half2` 向量化；
3. 支持 q/k packed layout；
4. 将 RoPE 与 q/k projection 后处理融合；
5. 将 RoPE 与 KV cache 写入融合；
6. 增加 batch 维 benchmark；
7. 补充锁频 benchmark 和多次运行波动分析。

建议下一步：

```text
短线：整理 RoPE 与 RMSNorm 的阶段性结果，形成 README 摘要。
中线：进入 INT8 Dequant-GEMV。
长线：回到 RoPE 做 FP16/half2 和融合版本。
```

---

## 10. 当前阶段结论

RoPE 已完成第二个算子的完整闭环：

```text
PyTorch Reference
    ↓
Naive CUDA
    ↓
Float4 CUDA
    ↓
Correctness Tests
    ↓
Benchmark CSV / Console Log
    ↓
Optimization Report
```

最终结论：

```text
CUDA Naive 相比 PyTorch Reference 有明显加速。
Float4 数值正确，相比 Naive 有稳定但较小的收益。
后续更有价值的优化方向是 FP16/half2 和与上下游算子融合。
```
