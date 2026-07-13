# Project Closure and Resume Notes

> 日期：2026-07-13  
> 状态：当前 CUDA 算子优化阶段封版

---

## 1. Closure Decision

EdgeLLM-KernelBench 当前阶段可以结束。

原因：

1. 已覆盖 RMSNorm、RoPE、INT8 Dequant-GEMV 三类 LLM 推理典型算子；
2. 每个算子都有 PyTorch reference、CUDA implementation、correctness tests、benchmark 和报告；
3. INT8 Dequant-GEMV 已完成多轮优化和负结果分析，不再停留在单一 happy path；
4. Nsight Systems 已补齐瓶颈量化，说明 rows=1 小 shape 受 launch / scheduling overhead 影响明显；
5. 继续新增 kernel 会进入新研究阶段，而不是当前项目的必要收尾。

---

## 2. Final Technical Scope

### RMSNorm

完成内容：

- PyTorch reference；
- CUDA naive；
- warp-level reduction；
- float4 vectorized path；
- correctness tests；
- benchmark report。

关键结论：

```text
Naive CUDA 相比 PyTorch reference 有稳定收益。
Float4 在大 rows 场景有明确收益，小 rows 场景与 warp 版本基本持平。
```

### RoPE

完成内容：

- PyTorch reference；
- CUDA naive fused kernel；
- float4 vectorized path；
- q/k dual tensor support；
- correctness tests；
- benchmark report。

关键结论：

```text
CUDA fused kernel 避免 PyTorch slicing / elementwise 组合开销，收益明显。
Float4 相比 naive 有稳定但较小收益。
```

### INT8 Dequant-GEMV

完成内容：

- PyTorch reference；
- CUDA naive fused dequant + GEMV；
- warp-level mapping；
- shared-memory x-tile experiment；
- float4 / char4 Vec4 path；
- FP16 activation support；
- half2 / char2 path；
- wide block experiment；
- Nsight Systems profiling；
- correctness tests；
- benchmark report。

关键结论：

```text
Fused dequant + GEMV 避免显式生成完整 FP32 dequant_weight。
Warp-level mapping 明显降低 block 调度压力。
X-tile 和 Wide 是有价值的负结果：同步和调度 tradeoff 抵消了潜在收益。
Vec4 在 FP32 rows=4 场景有效。
Half2 在 FP16 rows=4 场景有效。
rows=1 小 shape 受 launch / scheduling overhead 影响明显。
```

---

## 3. Validation Snapshot

最近全量测试：

```text
MAX_JOBS=2 PYTHONPATH=python /home/liujiayu/venvs/edge-llm-kernelbench/bin/python -m pytest -v
143 passed in 4.61s
```

Nsight 工具：

```text
Nsight Systems 2024.5.4
Nsight Compute 2024.3.1
```

Nsight Compute 限制：

```text
当前用户态权限无法采 hardware counter / occupancy。
已保留 ncu permission failure log。
```

---

## 4. Resume Version

中文简历描述：

```text
基于 NVIDIA Jetson Orin Nano 构建端侧 LLM CUDA 算子优化项目，覆盖 RMSNorm、RoPE、INT8 Dequant-GEMV 三类推理算子，完成 PyTorch reference、CUDA extension、正确性测试、benchmark 自动化与 Nsight profiling。针对 memory-bound 和小 batch 场景实现 warp-level reduction、float4/half2 向量化读取、fused dequant-GEMV 等优化，并系统记录 shared-memory tiling 和 wide-block 负结果，量化 rows=1 场景下 launch/scheduling overhead 对端到端收益的限制。
```

English resume bullets:

```text
Built an edge LLM CUDA kernel benchmark on Jetson Orin Nano, covering RMSNorm, RoPE, and INT8 Dequant-GEMV with PyTorch references, CUDA extensions, correctness tests, benchmark automation, and optimization reports.

Optimized fused CUDA kernels using warp-level reduction, vectorized global memory loads, fused dequantization, and FP16 half2 loading; achieved up to 22.4x over PyTorch reference for RoPE and 22.2x for INT8 Dequant-GEMV on tested shapes.

Profiled INT8 Dequant-GEMV with Nsight Systems and quantified launch/scheduling overhead versus kernel time, guiding the decision to stop pursuing wider block mappings and focus future work on rows=1 specialization or operator fusion.
```

---

## 5. Interview Talking Points

可重点展开：

1. 为什么 reference 慢：PyTorch composition 会产生多个 kernel launch 和中间 tensor；
2. 为什么 fused CUDA 快：减少 launch、中间读写和显式 dequant_weight；
3. 为什么 rows=1 难优化：kernel 本体很短，launch / scheduling overhead 占比高；
4. 为什么 X-tile 失败：shared memory 复用收益不足以覆盖同步开销；
5. 为什么 Wide 失败：减少 block 数不等于更快，512 threads/block 会带来 occupancy / scheduling tradeoff；
6. 为什么没有直接用 DP4A：当前 activation 是 FP32 / FP16，不是 packed INT8 activation；
7. 如何继续：INT8 activation + DP4A、rows=1 specialized mapping、operator fusion、TileLang / MXMACA 迁移。

---

## 6. Recommended Stop Point

当前项目不建议继续追加小型 CUDA 变体。

更合理的后续方式：

```text
当前仓库：作为 CUDA 算子优化项目封版。
新阶段 1：INT8 activation + DP4A。
新阶段 2：TileLang / MXMACA migration。
新阶段 3：接入真实小模型 decode graph 做 operator-level fusion profiling。
```
