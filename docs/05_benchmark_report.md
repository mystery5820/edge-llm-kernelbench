# EdgeLLM-KernelBench Benchmark Report

> 日期：2026-07-13  
> 平台：NVIDIA Jetson Orin Nano Super 8GB  
> 范围：RMSNorm、RoPE、INT8 Dequant-GEMV

---

## 1. 当前项目状态

本项目已经完成三个端侧大模型推理相关算子的阶段性闭环：

| 算子 | PyTorch Reference | CUDA Naive | CUDA Optimized | Tests | Benchmark | Report |
|---|---:|---:|---:|---:|---:|---:|
| RMSNorm | 已完成 | 已完成 | Warp / Float4 已完成 | 已完成 | 已完成 | 已完成 |
| RoPE | 已完成 | 已完成 | Float4 已完成 | 已完成 | 已完成 | 已完成 |
| INT8 Dequant-GEMV | 已完成 | 已完成 | Warp / Tiled / Vec4 / FP16 / Half2 已完成 | 已完成 | 已完成 | 已完成 |

最近全量回归：

```text
143 passed in 4.64s
```

---

## 2. 结果文件索引

### 2.1 RMSNorm

```text
results/rmsnorm_baseline_20260712_161636.csv
results/rmsnorm_warp_comparison_20260712_175730.csv
results/rmsnorm_float4_comparison_20260712_221121.csv
```

优化报告：

```text
docs/01_rmsnorm_optimization.md
```

### 2.2 RoPE

```text
results/rope_naive_comparison_20260712_230056.csv
results/rope_float4_comparison_20260712_231133.csv
```

优化报告：

```text
docs/02_rope_optimization.md
```

### 2.3 INT8 Dequant-GEMV

```text
results/int8_dequant_gemv_warp_comparison_20260713_122659.csv
results/int8_dequant_gemv_vec4_comparison_20260713_131217.csv
results/int8_dequant_gemv_fp32_vec4_comparison_20260713_135940.csv
results/int8_dequant_gemv_fp16_vec4_comparison_20260713_135528.csv
results/int8_dequant_gemv_fp16_half2_comparison_20260713_142951.csv
```

优化报告：

```text
docs/03_int8_dequant_gemv_optimization.md
```

---

## 3. RMSNorm Summary

实现版本：

```text
PyTorch Reference
CUDA Naive
CUDA Warp Shuffle
CUDA Float4
```

Float4 benchmark 参数：

```text
warmup=20
rounds=30
repeats=50
```

Float4 结果：

| rows | hidden | Float4 vs Reference | Float4 vs Naive | Float4 vs Warp |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 7.785x | 1.019x | 0.992x |
| 1 | 4096 | 7.877x | 1.016x | 0.998x |
| 16 | 4096 | 7.888x | 1.027x | 0.999x |
| 128 | 4096 | 4.588x | 1.061x | 1.049x |

结论：

```text
Naive CUDA 相比 PyTorch Reference 有稳定加速。
Warp Shuffle 数值正确，但没有形成稳定加速。
Float4 数值正确，在 rows=128, hidden=4096 场景有明确收益；
在小 rows 场景与 Warp 基本持平。
```

---

## 4. RoPE Summary

实现版本：

```text
PyTorch Reference
CUDA Naive
CUDA Float4
```

Float4 benchmark 参数：

```text
warmup=20
rounds=30
repeats=50
```

Float4 结果：

| seq_len | heads | head_dim | Float4 vs Reference | Float4 vs Naive |
|---:|---:|---:|---:|---:|
| 128 | 8 | 64 | 22.404x | 1.033x |
| 512 | 8 | 64 | 15.702x | 1.006x |
| 1024 | 16 | 64 | 7.593x | 1.006x |
| 2048 | 16 | 128 | 7.317x | 1.004x |

结论：

```text
CUDA Naive 将多个 PyTorch elementwise / slicing 操作融合到一个 kernel，
因此相比 PyTorch Reference 有明显收益。
Float4 相比 Naive 有稳定但较小的加速。
后续更有价值的方向是 FP16/half2 和与上下游操作融合。
```

---

## 5. INT8 Dequant-GEMV Summary

实现版本：

```text
PyTorch Reference
CUDA Naive
CUDA Warp-level
CUDA X-tile experiment
CUDA Vec4
CUDA FP16 activation
CUDA Half2
```

Vec4 benchmark 参数：

```text
warmup=5
rounds=10
repeats=10
```

Vec4 结果：

| rows | in_features | out_features | Vec4 vs Reference | Vec4 vs Naive | Vec4 vs Warp |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 22.169x | 2.797x | 1.063x |
| 1 | 2048 | 2048 | 19.108x | 2.599x | 0.984x |
| 4 | 2048 | 2048 | 9.206x | 4.604x | 1.430x |

FP16 activation 结果：

| rows | in_features | out_features | Half2 vs Reference | Half2 vs Naive | Half2 vs Warp |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 20.229x | 2.847x | 1.004x |
| 1 | 2048 | 2048 | 17.418x | 4.555x | 0.913x |
| 4 | 2048 | 2048 | 6.885x | 7.073x | 1.076x |

结论：

```text
CUDA Naive 避免显式生成完整 dequant_weight，中等规模下相比 PyTorch Reference 有明显收益。
Warp-level 版本通过一个 block 计算 8 个输出通道，减少 block 数并使用 warp shuffle 规约，
相对 Naive 获得 2.5x 到 3.2x 的加速。
X-tile shared-memory 复用实验数值正确，但同步开销超过 x 复用收益。
Vec4 版本在 rows=4 的场景明显快于 Warp，在 rows=1 场景与 Warp 基本持平。
FP16 activation 已完成基础功能支持，内部仍使用 FP32 accumulation。
Half2 版本使用 half2/char2 向量化读取，再转为 FP32 乘加以保持数值契约；
rows=4 场景有小幅收益，rows=1 场景没有稳定领先。
DP4A 不能直接用于 FP32/FP16 activation 主乘加路径；需要 INT8 activation 路径后再评估。
```

---

## 6. Cross-Operator Observations

### 6.1 Kernel fusion usually beats PyTorch composition

RoPE 和 INT8 Dequant-GEMV 的收益很明显，核心原因是自定义 CUDA kernel 避免了多个 PyTorch op 组合带来的中间 tensor 和多次 kernel launch。

### 6.2 Vectorization is not always a large win

RMSNorm Float4、RoPE Float4 和 INT8 GEMV Vec4 都是数值正确且有收益的优化，但收益幅度依赖场景：

- 小算子会被 launch overhead 稀释；
- 如果 Naive 访存已经较连续，float4 的额外收益会收敛；
- rows / seq_len / out_features 越能提供并行度，优化越容易体现。

### 6.3 Warp-level mapping can strongly reduce scheduling overhead

INT8 Dequant-GEMV 的 warp-level 版本比 naive 明显更快，因为它把：

```text
一个 block 一个输出元素
```

改成：

```text
一个 block 八个输出元素
```

这比单纯减少 shared memory 更直接地降低了 block 调度压力。

---

## 7. Current Limitations

当前项目仍有以下限制：

1. CUDA kernel 多数仍以 FP32 为主；
2. RMSNorm / RoPE 尚未完成 FP16/half2 版本，INT8 GEMV half2 收益仍有限；
3. INT8 GEMV 尚未使用 DP4A，因为还没有 INT8 activation 路径；
4. benchmark 尚未统一锁频和温度记录；
5. 没有 Nsight Compute / Nsight Systems profiling 数据；
6. 没有 TileLang / MXMACA 迁移实现。

---

## 8. Recommended Next Steps

短线：

```text
补充 README 总览表和运行命令；
为 benchmark 增加统一 metadata；
整理一份 GitHub 展示用项目说明。
```

中线：

```text
RMSNorm FP16 / half2；
RoPE half2；
INT8 Dequant-GEMV rows=1 专门 kernel；
INT8 Dequant-GEMV 更多 output channel / block；
INT8 activation 路径与 DP4A；
INT8 GEMV rows=1 专门 kernel。
```

长线：

```text
迁移一个算子到 TileLang；
整理 CUDA 到 MXMACA / 国产 GPU 的迁移笔记；
补 Nsight profiling 截图和瓶颈分析。
```

---

## 9. Overall Conclusion

当前项目已经达到阶段性目标：

```text
三个典型大模型推理算子
    ↓
PyTorch Reference
    ↓
CUDA Baseline
    ↓
CUDA Optimized
    ↓
Correctness Tests
    ↓
Benchmark Results
    ↓
Optimization Reports
```

这说明项目已经不只是“跑通 CUDA kernel”，而是具备了较完整的算子开发、验证、benchmark 和性能分析闭环。
