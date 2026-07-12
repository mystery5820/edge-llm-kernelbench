# EdgeLLM-KernelBench：面向端侧大模型推理的 GPU 算子优化实践平台

## 1. 项目概述

**EdgeLLM-KernelBench** 是一个面向端侧大模型推理场景的 GPU 算子开发与性能优化实践项目。项目基于 NVIDIA Jetson Orin Nano Super 8GB 开发板，围绕大模型推理中常见且具有代表性的核心算子，完成从 PyTorch baseline、CUDA kernel 实现、正确性验证、性能 Benchmark、优化分析到文档总结的完整闭环。

项目重点关注以下几类大模型推理中的基础算子：

1. RMSNorm：大语言模型中常见的归一化算子；
2. RoPE：旋转位置编码算子，常用于 Transformer Attention 结构；
3. INT8 Dequant-GEMV：面向量化大模型推理的反量化与矩阵向量乘融合算子。

该项目不是一个普通的 AI 应用项目，而是一个偏底层、偏性能优化、偏 GPU 编程训练的开源实践项目。项目目标是通过 Jetson 平台完成 CUDA 算子开发训练，建立 GPU 算子开发的基本能力，并为后续迁移到国产 GPU 软件栈、TileLang、MXMACA 等方向做准备。

---

## 2. 项目背景

近年来，大模型推理部署已经从单纯的模型调用，逐渐进入到推理框架、算子优化、量化压缩、异构硬件适配和系统服务化协同优化的阶段。

在端侧 AI 场景中，模型部署通常受到以下因素限制：

1. 显存容量有限；
2. 算力资源有限；
3. 内存带宽有限；
4. batch size 通常较小；
5. 低延迟要求明显；
6. 模型量化和算子融合需求强烈。

在大语言模型推理过程中，除了 Attention 和 GEMM/GEMV 等核心计算外，RMSNorm、RoPE、反量化、激活函数、小规模 reduction 等算子也会频繁出现。这些算子单独看计算量可能不大，但在端侧推理中，由于 kernel launch overhead、访存开销和中间 tensor 读写，它们会对整体推理效率产生明显影响。

因此，本项目希望从几个具有代表性的基础算子入手，逐步理解 GPU 算子开发中的关键问题：

1. 如何设计 thread/block/grid；
2. 如何进行全局内存访问优化；
3. 如何使用 shared memory；
4. 如何进行 block-level reduction；
5. 如何进行 warp-level reduction；
6. 如何减少中间 tensor；
7. 如何通过算子融合减少访存和 kernel launch；
8. 如何进行正确性验证；
9. 如何进行性能测试和结果分析；
10. 如何把 CUDA 上的优化思路迁移到其他 GPU 编程模型。

---

## 3. 项目定位

本项目的定位是：

> 面向端侧大模型推理的 GPU 算子开发与优化实践平台。

它的核心不是做一个完整的大模型推理框架，也不是单纯跑通某个模型，而是聚焦于“大模型推理中典型算子的实现、验证和优化”。

项目重点体现以下能力：

1. GPU 编程基础能力；
2. CUDA kernel 编写能力；
3. 算子正确性验证能力；
4. Benchmark 设计能力；
5. 性能瓶颈分析能力；
6. 算子优化迭代能力；
7. 面向国产 GPU 软件栈的迁移思考能力。

---

## 4. 与已有 RK3588 项目的关系

用户目前已有一个 RK3588 端侧推理服务项目，主要方向包括：

1. RK3588 板端模型部署；
2. RKNN 视觉模型推理；
3. RKLLM 大语言模型推理；
4. OpenAI-like API 服务；
5. 模型注册表；
6. worker 后端；
7. metrics 指标；
8. 服务稳定性；
9. 文档和 demo 脚本。

该项目已经体现了较强的端侧 AI 工程能力，但更多集中在“模型部署、推理服务、框架集成、工程化封装”层面。

EdgeLLM-KernelBench 则用于补齐另一块能力：

> 从“会部署模型”进一步扩展到“理解底层算子、会做 GPU kernel 开发和性能优化”。

两者形成互补：

| 项目 | 主要能力 | 技术层次 |
|---|---|---|
| EdgeInfer-RK3588 | 模型部署、推理服务、API、worker、metrics | 推理系统层 |
| EdgeLLM-KernelBench | CUDA kernel、算子实现、性能优化、Benchmark | 算子与硬件执行层 |

最终可以形成一条更完整的技术主线：

```text
嵌入式 AI 部署
    ↓
RK3588 NPU 推理服务
    ↓
Jetson GPU 算子优化
    ↓
国产 GPU / TileLang / MXMACA 算子开发
```

---

## 5. 项目目标

### 5.1 总体目标

基于 Jetson Orin Nano Super 8GB，实现一个可复现、可测试、可 Benchmark、可文档化的 GPU 算子优化项目，重点完成 RMSNorm、RoPE、INT8 Dequant-GEMV 三类算子的 CUDA 实现与性能分析。

### 5.2 具体目标

项目计划实现以下目标：

1. 搭建 Jetson CUDA 开发环境；
2. 建立统一的项目目录结构；
3. 实现 PyTorch reference baseline；
4. 实现 CUDA naive kernel；
5. 实现 CUDA optimized kernel；
6. 对比 PyTorch、CUDA naive、CUDA optimized 的性能；
7. 对比不同输入规模下的性能表现；
8. 输出 benchmark CSV；
9. 输出 correctness report；
10. 输出优化分析文档；
11. 总结 CUDA 到 TileLang/MXMACA 的迁移思路；
12. 形成可展示的 GitHub 开源项目；
13. 形成可写入简历和夏令营报名材料的项目经历。

---

## 6. 项目非目标

为了保证项目在短时间内可落地，本项目暂时不做以下内容：

1. 不实现完整大模型推理框架；
2. 不实现完整 Transformer；
3. 不从零实现 FlashAttention；
4. 不追求超过 TensorRT/cuBLAS 等工业库；
5. 不做复杂多 GPU 通信；
6. 不做训练加速；
7. 不做完整量化工具链；
8. 不做模型转换器；
9. 不做端到端聊天服务；
10. 不把 Jetson 项目包装成产品级推理服务。

本项目的重点是：

> 小而完整，真实可跑，有正确性验证，有性能数据，有优化分析，有迁移思考。

---

## 7. 硬件平台

### 7.1 开发板

项目使用的硬件平台：

```text
NVIDIA Jetson Orin Nano Super 8GB
```

该平台适合作为 GPU 算子开发入门和端侧推理性能实验平台。相比桌面级 GPU，它的算力和显存规模更小，但这恰好适合端侧推理场景下的性能优化训练。

### 7.2 平台特点

Jetson 平台适合本项目的原因：

1. 支持 CUDA 编程；
2. 支持 PyTorch GPU 版本；
3. 支持端侧部署场景；
4. 功耗和资源受限，更接近边缘计算设备；
5. 可以进行真实硬件上的 kernel benchmark；
6. 适合训练 GPU 算子开发、调试和性能分析能力。

### 7.3 注意事项

本项目使用 Jetson/CUDA 作为训练平台，但目标不是证明项目适配某一个特定国产 GPU，而是通过 CUDA 掌握 GPU 算子开发的通用思想。

后续可以将该项目中的算子设计、Benchmark 方法和优化思路迁移到：

1. TileLang；
2. MXMACA；
3. 国产 GPU 平台；
4. 其他异构计算平台。

---

## 8. 软件环境

### 8.1 基础环境

建议环境如下：

```text
操作系统：Jetson Linux / Ubuntu
开发语言：C++ / CUDA / Python
Python：3.8+
深度学习框架：PyTorch
构建工具：CMake / setuptools / torch.utils.cpp_extension
性能测试：Python benchmark + CUDA event
结果存储：CSV / Markdown
版本管理：Git + GitHub
```

### 8.2 推荐依赖

```text
Python packages:
- torch
- numpy
- pandas
- matplotlib
- pytest
- tabulate

System packages:
- build-essential
- cmake
- git
- nvcc
```

### 8.3 开发方式

建议优先使用 PyTorch C++/CUDA Extension 的方式开发 kernel。这样可以方便地在 Python 中调用 CUDA 算子，并与 PyTorch reference 进行对比。

典型流程：

```text
Python test script
    ↓
PyTorch reference output
    ↓
CUDA extension output
    ↓
误差对比
    ↓
性能 Benchmark
    ↓
CSV 结果记录
```

---

## 9. 技术路线

本项目采用“三层实现 + 两类验证 + 一套文档”的路线。

### 9.1 三层实现

每个算子尽量包含三类实现：

1. PyTorch reference 实现；
2. CUDA naive 实现；
3. CUDA optimized 实现。

例如 RMSNorm：

```text
rmsnorm_torch()
rmsnorm_cuda_naive()
rmsnorm_cuda_optimized()
```

### 9.2 两类验证

每个算子都需要进行两类验证：

1. 正确性验证；
2. 性能验证。

正确性验证关注：

```text
max_abs_error
mean_abs_error
relative_error
是否满足误差阈值
```

性能验证关注：

```text
平均耗时
最小耗时
最大耗时
吞吐量
加速比
不同输入规模下的变化趋势
```

### 9.3 一套文档

每个算子都需要有独立文档：

```text
docs/01_rmsnorm_optimization.md
docs/02_rope_kernel.md
docs/03_int8_dequant_gemv.md
```

每份文档需要说明：

1. 算子背景；
2. 数学定义；
3. 输入输出；
4. naive 实现思路；
5. optimized 实现思路；
6. 正确性验证；
7. 性能结果；
8. 当前瓶颈；
9. 后续优化方向。

---

## 10. 仓库结构设计

建议仓库名称：

```text
edge-llm-kernelbench
```

建议目录结构：

```text
edge-llm-kernelbench/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── CMakeLists.txt
│
├── docs/
│   ├── project_design.md
│   ├── 00_project_overview.md
│   ├── 01_rmsnorm_optimization.md
│   ├── 02_rope_kernel.md
│   ├── 03_int8_dequant_gemv.md
│   ├── 04_cuda_to_tilelang_mxmaca_notes.md
│   └── 05_benchmark_report.md
│
├── kernels/
│   ├── rmsnorm/
│   │   ├── rmsnorm.cpp
│   │   ├── rmsnorm_kernel.cu
│   │   └── rmsnorm.h
│   │
│   ├── rope/
│   │   ├── rope.cpp
│   │   ├── rope_kernel.cu
│   │   └── rope.h
│   │
│   └── int8_dequant_gemv/
│       ├── int8_gemv.cpp
│       ├── int8_gemv_kernel.cu
│       └── int8_gemv.h
│
├── python/
│   └── edge_kernelbench/
│       ├── __init__.py
│       ├── rmsnorm.py
│       ├── rope.py
│       └── int8_gemv.py
│
├── tests/
│   ├── test_rmsnorm.py
│   ├── test_rope.py
│   └── test_int8_gemv.py
│
├── benchmarks/
│   ├── bench_rmsnorm.py
│   ├── bench_rope.py
│   ├── bench_int8_gemv.py
│   └── run_all_benchmarks.py
│
├── scripts/
│   ├── setup_jetson.sh
│   ├── build_extension.sh
│   ├── run_all_tests.sh
│   └── run_all_benchmarks.sh
│
└── results/
    └── jetson_orin_nano_super/
        ├── rmsnorm.csv
        ├── rope.csv
        ├── int8_gemv.csv
        └── summary.md
```

---

## 11. 核心算子一：RMSNorm

### 11.1 算子背景

RMSNorm 是大语言模型中常见的归一化算子。相比 LayerNorm，RMSNorm 不减去均值，只根据输入向量的均方根进行缩放。

在很多 LLM 中，RMSNorm 会出现在 Transformer block 的前后，例如：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

由于每一层都会调用 RMSNorm，因此它是一个频繁出现的小算子。

### 11.2 数学定义

给定输入向量：

```text
x = [x1, x2, ..., xn]
```

RMSNorm 计算过程为：

```text
rms = sqrt((x1^2 + x2^2 + ... + xn^2) / n + eps)

y_i = x_i / rms * weight_i
```

其中：

```text
x: 输入张量
weight: 可学习缩放参数
eps: 防止除零的小常数
y: 输出张量
```

### 11.3 输入输出

典型输入输出：

```text
input:  [batch_size, hidden_size]
weight: [hidden_size]
output: [batch_size, hidden_size]
```

测试规模：

```text
batch_size: 1, 2, 4, 8
hidden_size: 512, 1024, 2048, 4096
dtype: fp32, fp16
```

### 11.4 实现版本

计划实现三个版本：

#### 版本 A：PyTorch Reference

使用 PyTorch 实现 RMSNorm，用于正确性对比。

```text
torch_rmsnorm(x, weight, eps)
```

#### 版本 B：CUDA Naive

每一行使用一个 block，block 内线程计算平方和，然后进行 reduction。

特点：

1. 逻辑清晰；
2. 易于验证；
3. 性能不一定最优；
4. 适合作为第一个 CUDA kernel。

#### 版本 C：CUDA Optimized

优化方向：

1. shared memory reduction；
2. warp-level reduction；
3. 向量化读取；
4. 减少同步；
5. 优化 block size；
6. 支持 half 数据类型。

### 11.5 正确性验证

验证指标：

```text
max_abs_error
mean_abs_error
relative_error
```

误差阈值建议：

```text
fp32: max_abs_error < 1e-4
fp16: max_abs_error < 1e-2
```

### 11.6 Benchmark 指标

记录字段：

```text
operator
dtype
batch_size
hidden_size
implementation
latency_ms
speedup_vs_torch
max_abs_error
mean_abs_error
```

示例：

```text
rmsnorm,fp32,1,4096,torch,0.120,1.00,0,0
rmsnorm,fp32,1,4096,cuda_naive,0.080,1.50,1e-5,3e-6
rmsnorm,fp32,1,4096,cuda_optimized,0.045,2.67,1e-5,3e-6
```

### 11.7 预期收获

通过 RMSNorm，可以掌握：

1. reduction 算子基本写法；
2. shared memory 使用；
3. block 内并行规约；
4. warp-level primitive；
5. 小算子的性能测试方法；
6. PyTorch extension 基本流程。

---

## 12. 核心算子二：RoPE

### 12.1 算子背景

RoPE，全称 Rotary Position Embedding，即旋转位置编码，是 Transformer 大模型中常见的位置编码方法。它通常作用于 Attention 中的 query 和 key。

RoPE 的特点是通过对 query/key 的偶数维和奇数维进行旋转变换，把位置信息注入到 Attention 计算中。

### 12.2 数学定义

对于一对维度：

```text
x_even
x_odd
```

RoPE 计算为：

```text
y_even = x_even * cos - x_odd * sin
y_odd  = x_even * sin + x_odd * cos
```

其中 cos 和 sin 与 token 位置有关。

### 12.3 输入输出

典型输入：

```text
q:   [seq_len, num_heads, head_dim]
k:   [seq_len, num_heads, head_dim]
cos: [seq_len, head_dim / 2]
sin: [seq_len, head_dim / 2]
```

输出：

```text
q_out: [seq_len, num_heads, head_dim]
k_out: [seq_len, num_heads, head_dim]
```

测试规模：

```text
seq_len: 128, 512, 1024, 2048
num_heads: 8, 16, 32
head_dim: 64, 128
dtype: fp32, fp16
```

### 12.4 实现版本

计划实现：

#### 版本 A：PyTorch Reference

使用 PyTorch indexing 实现 RoPE，用于正确性对比。

#### 版本 B：CUDA Naive

每个线程处理一组 even/odd 维度。

特点：

1. 映射关系清楚；
2. 适合练习多维 index 展平；
3. 易于验证。

#### 版本 C：CUDA Optimized

优化方向：

1. q 和 k 合并处理；
2. 连续访存；
3. half2 向量化；
4. 减少重复读取 cos/sin；
5. 尽量避免不合并访存。

### 12.5 正确性验证

验证方式：

```text
torch_q_out vs cuda_q_out
torch_k_out vs cuda_k_out
```

指标：

```text
max_abs_error
mean_abs_error
```

误差阈值：

```text
fp32: max_abs_error < 1e-4
fp16: max_abs_error < 1e-2
```

### 12.6 Benchmark 指标

记录字段：

```text
operator
dtype
seq_len
num_heads
head_dim
implementation
latency_ms
speedup_vs_torch
max_abs_error
mean_abs_error
```

### 12.7 预期收获

通过 RoPE，可以掌握：

1. elementwise GPU kernel；
2. 多维张量 index 映射；
3. 连续访存优化；
4. q/k 双输入处理；
5. half/half2 初步优化；
6. LLM Attention 前处理算子的实现方式。

---

## 13. 核心算子三：INT8 Dequant-GEMV

### 13.1 算子背景

大模型端侧推理中，量化是降低显存占用和提升推理效率的重要手段。常见做法是将权重量化为 INT8 或更低 bit 格式，在推理时进行反量化并参与矩阵乘计算。

对于 decode 阶段，batch size 通常较小，线性层计算可以近似看作 GEMV：

```text
y = W x
```

其中：

```text
W: [out_features, in_features]
x: [in_features]
y: [out_features]
```

如果 W 是 INT8，则需要：

```text
W_int8 -> dequant -> GEMV
```

如果先把 W_int8 完整反量化成 FP16/FP32，再调用 GEMV，会产生额外显存读写。因此可以尝试融合：

```text
fused_dequant_gemv(W_int8, scale, x) -> y
```

### 13.2 数学定义

权重反量化：

```text
W_fp = W_int8 * scale
```

矩阵向量乘：

```text
y_i = sum_j W_fp[i, j] * x_j
```

融合后：

```text
y_i = sum_j (W_int8[i, j] * scale_i) * x_j
```

其中 scale 可以是：

```text
per-tensor scale
per-channel scale
```

本项目优先实现 per-channel scale。

### 13.3 输入输出

输入：

```text
W_int8: [out_features, in_features]
scale:  [out_features]
x:      [in_features]
```

输出：

```text
y:      [out_features]
```

测试规模：

```text
in_features: 512, 1024, 2048, 4096
out_features: 512, 1024, 2048, 4096
dtype:
  W_int8: int8
  x: fp16/fp32
  y: fp16/fp32
```

### 13.4 实现版本

计划实现四个版本：

#### 版本 A：PyTorch Reference

使用 PyTorch 完成：

```text
W_fp = W_int8.float() * scale[:, None]
y = W_fp @ x
```

#### 版本 B：Separate Dequant + GEMV

先使用一个 kernel 完成 dequant：

```text
W_int8 -> W_fp
```

再使用 PyTorch 或自定义 CUDA GEMV 完成计算。

该版本用于证明“中间 tensor 写回显存”的开销。

#### 版本 C：CUDA Naive Fused Dequant-GEMV

一个 CUDA kernel 内完成：

```text
读取 W_int8
读取 scale
读取 x
反量化
累加
输出 y
```

特点：

1. 减少中间 tensor；
2. 降低显存写回；
3. 更贴近真实量化推理；
4. 是项目亮点之一。

#### 版本 D：CUDA Optimized Fused Dequant-GEMV

优化方向：

1. 每个 block 计算一个或多个 out_features；
2. block 内 reduction；
3. shared memory 缓存 x；
4. 向量化读取 int8；
5. 尽量提升访存连续性；
6. 对不同 in_features/out_features 调参。

### 13.5 正确性验证

验证指标：

```text
max_abs_error
mean_abs_error
relative_error
```

误差阈值：

```text
fp32 accumulation: max_abs_error < 1e-3
fp16 accumulation: max_abs_error < 1e-1
```

### 13.6 Benchmark 指标

记录字段：

```text
operator
dtype
in_features
out_features
implementation
latency_ms
speedup_vs_torch
speedup_vs_separate
max_abs_error
mean_abs_error
```

示例：

```text
int8_dequant_gemv,fp32,4096,4096,torch,2.50,1.00,1.00,0,0
int8_dequant_gemv,fp32,4096,4096,separate,1.80,1.39,1.00,1e-3,2e-4
int8_dequant_gemv,fp32,4096,4096,fused_naive,1.30,1.92,1.38,1e-3,2e-4
int8_dequant_gemv,fp32,4096,4096,fused_optimized,0.95,2.63,1.89,1e-3,2e-4
```

### 13.7 预期收获

通过 INT8 Dequant-GEMV，可以掌握：

1. 量化推理基本思想；
2. 反量化计算；
3. GEMV 并行化；
4. 算子融合；
5. 显存读写优化；
6. 小 batch LLM decode 场景下的性能瓶颈；
7. 与 Dequant GEMM、TileLang kernel 的关联。

---

## 14. Benchmark 设计

### 14.1 Benchmark 原则

Benchmark 需要满足以下原则：

1. 每个实现先 warmup；
2. 使用 CUDA event 计时；
3. 每组测试重复多次；
4. 输出平均值、最小值、最大值；
5. 避免只测一次；
6. 保证输入规模一致；
7. 保证测试前后同步；
8. 记录硬件和软件环境；
9. 结果输出为 CSV；
10. 汇总结果写入 Markdown 报告。

### 14.2 Benchmark 流程

每个 benchmark 脚本执行流程：

```text
1. 生成输入数据
2. 执行 PyTorch reference
3. 执行 CUDA 实现
4. 比较正确性
5. warmup
6. 正式计时
7. 统计 latency
8. 计算 speedup
9. 写入 CSV
10. 打印 summary
```

### 14.3 Benchmark 输出字段

统一字段：

```text
timestamp
device
operator
implementation
dtype
shape
latency_mean_ms
latency_min_ms
latency_max_ms
speedup
max_abs_error
mean_abs_error
passed
```

### 14.4 结果展示

在 README 中展示简化表格：

```text
| Operator | Shape | Baseline | Optimized | Speedup | Error |
|---|---:|---:|---:|---:|---:|
| RMSNorm | B=1,H=4096 | 0.120 ms | 0.045 ms | 2.67x | 1e-5 |
| RoPE | S=1024,H=32,D=128 | 0.300 ms | 0.180 ms | 1.67x | 1e-5 |
| INT8 Dequant-GEMV | 4096x4096 | 2.50 ms | 0.95 ms | 2.63x | 1e-3 |
```

真实数据以后以 Jetson 实测为准，README 中不要写虚假性能数据。项目初期可以写：

```text
Benchmark results are measured on Jetson Orin Nano Super 8GB. Numbers will be updated after each optimization stage.
```

---

## 15. 正确性验证设计

### 15.1 验证目标

正确性验证用于确保 CUDA kernel 的输出与 PyTorch reference 保持一致。

### 15.2 验证指标

主要指标：

```text
max_abs_error = max(abs(y_cuda - y_ref))
mean_abs_error = mean(abs(y_cuda - y_ref))
relative_error = max_abs_error / max(abs(y_ref))
```

### 15.3 验证输入

每个算子应覆盖多种输入规模：

```text
小规模：方便调试
中规模：接近真实模型
大规模：观察性能瓶颈
```

### 15.4 测试脚本

使用 pytest 组织测试：

```text
pytest tests/test_rmsnorm.py
pytest tests/test_rope.py
pytest tests/test_int8_gemv.py
```

### 15.5 通过标准

每个算子的测试输出：

```text
[PASS] RMSNorm fp32 B=1 H=4096 max_error=...
[PASS] RoPE fp16 S=1024 H=32 D=128 max_error=...
[PASS] INT8 Dequant-GEMV fp32 M=4096 N=4096 max_error=...
```

---

## 16. 优化方法设计

### 16.1 通用优化方向

项目中会重点尝试以下 GPU kernel 优化方法：

1. 合理划分 grid/block；
2. 尽量保证 global memory coalescing；
3. 使用 shared memory 缓存重复访问的数据；
4. 使用 block-level reduction；
5. 使用 warp-level reduction；
6. 减少 thread divergence；
7. 减少不必要的同步；
8. 减少中间 tensor；
9. 通过算子融合降低访存；
10. 针对不同 shape 调整 block size；
11. 尝试 fp16/half2 优化；
12. 使用 CUDA event 进行稳定计时。

### 16.2 RMSNorm 优化重点

RMSNorm 的瓶颈主要在 reduction 和访存。

优化重点：

1. 每个 block 处理一行；
2. 每个线程处理多个元素；
3. 使用 shared memory 做平方和规约；
4. 使用 warp-level reduction 减少同步；
5. 对 hidden_size 做调参；
6. 对 fp16 做向量化读取。

### 16.3 RoPE 优化重点

RoPE 的瓶颈主要在 index 映射和内存访问。

优化重点：

1. 每个线程处理一对 even/odd；
2. 保证 q/k 访问连续；
3. 减少 cos/sin 重复读取；
4. q 和 k 可以合并处理；
5. fp16 下尝试 half2。

### 16.4 INT8 Dequant-GEMV 优化重点

INT8 Dequant-GEMV 的瓶颈主要在访存和 reduction。

优化重点：

1. 融合 dequant 和 GEMV；
2. 避免生成中间 W_fp tensor；
3. 使用 shared memory 缓存 x；
4. 每个 block 负责一个输出通道或多个输出通道；
5. 对 int8 权重进行连续读取；
6. 研究 per-channel scale 的读取方式；
7. 对不同矩阵规模调参。

---

## 17. 文档设计

项目文档需要体现“不是只写代码，而是理解问题、分析问题、优化问题”。

### 17.1 README.md

README 需要包含：

1. 项目简介；
2. 项目背景；
3. 硬件平台；
4. 已实现算子；
5. 快速开始；
6. 编译方法；
7. 测试方法；
8. Benchmark 方法；
9. 当前结果；
10. 优化总结；
11. 后续计划；
12. 与国产 GPU 软件栈的迁移思考。

### 17.2 算子文档

每个算子一份独立文档：

```text
docs/01_rmsnorm_optimization.md
docs/02_rope_kernel.md
docs/03_int8_dequant_gemv.md
```

每份文档包含：

1. 算子作用；
2. 数学公式；
3. 输入输出；
4. baseline 实现；
5. CUDA naive 实现；
6. CUDA optimized 实现；
7. 正确性验证；
8. Benchmark 数据；
9. 性能瓶颈；
10. 后续优化方向。

### 17.3 迁移思考文档

新增：

```text
docs/04_cuda_to_tilelang_mxmaca_notes.md
```

内容包括：

1. CUDA kernel 的基本结构；
2. TileLang 的算子表达思路；
3. MXMACA 与 CUDA 的迁移关系；
4. 哪些优化思想是通用的；
5. 哪些实现细节需要根据硬件调整；
6. 后续如何把 RMSNorm、RoPE、INT8 Dequant-GEMV 迁移到 TileLang。

---

## 18. 与 TileLang / MXMACA 的迁移思考

虽然本项目在 Jetson/CUDA 上开发，但设计时需要主动考虑后续迁移到国产 GPU 软件栈。

### 18.1 可迁移的思想

以下思想具有较强通用性：

1. 算子输入输出定义；
2. 正确性验证方式；
3. Benchmark 设计方式；
4. shape sweep 方法；
5. reduction 优化思路；
6. shared memory / local memory 缓存思想；
7. 算子融合思想；
8. 访存连续性分析；
9. fp16/int8 数据类型处理；
10. 性能报告写法。

### 18.2 需要适配的部分

以下部分在迁移时需要根据国产 GPU 软件栈重新实现：

1. kernel 编写语法；
2. thread/block 映射方式；
3. memory hierarchy 细节；
4. warp/wavefront 等执行单元差异；
5. 编译工具链；
6. profiler 工具；
7. intrinsic 函数；
8. half/int8 向量化方式；
9. 运行时 API；
10. 性能调参策略。

### 18.3 迁移路线

后续迁移路线可以设计为：

```text
CUDA PyTorch Extension
    ↓
抽象算子输入输出和测试数据
    ↓
整理 benchmark 与 correctness 框架
    ↓
使用 TileLang 重写 RMSNorm
    ↓
使用 TileLang 重写 RoPE
    ↓
使用 TileLang 重写 INT8 Dequant-GEMV
    ↓
在国产 GPU / MXMACA 环境中验证
```

### 18.4 报名材料中的表达方式

可以这样描述：

```text
本项目当前基于 Jetson/CUDA 平台完成 GPU 算子开发训练，重点掌握线程映射、访存优化、规约、算子融合和 Benchmark 方法。项目设计时保留了清晰的 reference、kernel、test、benchmark 分层结构，便于后续将 RMSNorm、RoPE、INT8 Dequant-GEMV 等算子迁移到 TileLang/MXMACA 等国产 GPU 软件栈中进行进一步验证。
```

---

## 19. 开发计划

### 19.1 第一阶段：项目骨架与环境验证

目标：

1. 确认 Jetson CUDA 可用；
2. 确认 PyTorch CUDA 可用；
3. 创建项目仓库；
4. 建立目录结构；
5. 编写 README 初版；
6. 跑通 CUDA extension 示例。

交付物：

```text
README.md
requirements.txt
scripts/setup_jetson.sh
scripts/build_extension.sh
docs/project_design.md
```

完成标准：

```text
python -c "import torch; print(torch.cuda.is_available())"
```

输出为：

```text
True
```

并且可以成功编译一个最小 CUDA extension。

---

### 19.2 第二阶段：RMSNorm 实现与优化

目标：

1. 实现 PyTorch RMSNorm；
2. 实现 CUDA naive RMSNorm；
3. 实现 CUDA optimized RMSNorm；
4. 完成 correctness test；
5. 完成 benchmark；
6. 编写 RMSNorm 优化文档。

交付物：

```text
kernels/rmsnorm/
tests/test_rmsnorm.py
benchmarks/bench_rmsnorm.py
docs/01_rmsnorm_optimization.md
results/jetson_orin_nano_super/rmsnorm.csv
```

完成标准：

```text
pytest tests/test_rmsnorm.py
python benchmarks/bench_rmsnorm.py
```

均可正常运行，并输出 CSV 结果。

---

### 19.3 第三阶段：RoPE 实现与优化

目标：

1. 实现 PyTorch RoPE；
2. 实现 CUDA RoPE；
3. 支持 q/k 输入；
4. 完成 correctness test；
5. 完成 benchmark；
6. 编写 RoPE 文档。

交付物：

```text
kernels/rope/
tests/test_rope.py
benchmarks/bench_rope.py
docs/02_rope_kernel.md
results/jetson_orin_nano_super/rope.csv
```

完成标准：

```text
pytest tests/test_rope.py
python benchmarks/bench_rope.py
```

均可正常运行，并输出 CSV 结果。

---

### 19.4 第四阶段：INT8 Dequant-GEMV 实现与优化

目标：

1. 实现 PyTorch reference；
2. 实现 separate dequant + GEMV；
3. 实现 fused dequant-gemv；
4. 实现 optimized fused dequant-gemv；
5. 完成 correctness test；
6. 完成 benchmark；
7. 编写 INT8 Dequant-GEMV 文档。

交付物：

```text
kernels/int8_dequant_gemv/
tests/test_int8_gemv.py
benchmarks/bench_int8_gemv.py
docs/03_int8_dequant_gemv.md
results/jetson_orin_nano_super/int8_gemv.csv
```

完成标准：

```text
pytest tests/test_int8_gemv.py
python benchmarks/bench_int8_gemv.py
```

均可正常运行，并输出 CSV 结果。

---

### 19.5 第五阶段：结果整理与报名材料包装

目标：

1. 汇总 benchmark 结果；
2. 编写 summary 文档；
3. 更新 README；
4. 补充项目截图；
5. 编写简历项目描述；
6. 编写夏令营报名材料中的项目说明；
7. 整理后续计划。

交付物：

```text
docs/05_benchmark_report.md
docs/04_cuda_to_tilelang_mxmaca_notes.md
results/jetson_orin_nano_super/summary.md
resume_project_description.md
README.md
```

完成标准：

1. GitHub 仓库首页清晰；
2. 能看到完整项目背景；
3. 能看到已实现算子；
4. 能看到测试命令；
5. 能看到 benchmark 结果；
6. 能看到优化分析；
7. 能看到后续迁移 TileLang/MXMACA 的计划。

---

## 20. 时间安排建议

考虑到报名时间较紧，建议采用 7 天冲刺方式。

### Day 1：环境和仓库

任务：

1. 确认 Jetson 系统；
2. 确认 CUDA；
3. 确认 PyTorch；
4. 建立仓库；
5. 写 README 初版；
6. 跑通 CUDA extension demo。

产出：

```text
项目骨架
README 初版
环境验证截图
```

### Day 2：RMSNorm naive

任务：

1. 写 PyTorch RMSNorm；
2. 写 CUDA naive RMSNorm；
3. 写 correctness test；
4. 写 benchmark 初版。

产出：

```text
RMSNorm 可运行版本
```

### Day 3：RMSNorm optimized

任务：

1. 加 shared memory reduction；
2. 加 warp-level reduction；
3. 测试不同 hidden size；
4. 写 RMSNorm 文档。

产出：

```text
RMSNorm 优化文档
RMSNorm benchmark 数据
```

### Day 4：RoPE

任务：

1. 写 PyTorch RoPE；
2. 写 CUDA RoPE；
3. 测试 q/k；
4. benchmark；
5. 写 RoPE 文档。

产出：

```text
RoPE 可运行版本
RoPE benchmark 数据
```

### Day 5：INT8 Dequant-GEMV naive

任务：

1. 写 PyTorch reference；
2. 写 int8 权重生成；
3. 写 separate dequant；
4. 写 naive fused dequant-gemv；
5. correctness test。

产出：

```text
INT8 Dequant-GEMV 初版
```

### Day 6：INT8 Dequant-GEMV optimized

任务：

1. 优化 fused kernel；
2. 加 shared memory；
3. 对比 separate 和 fused；
4. 生成 benchmark；
5. 写 INT8 文档。

产出：

```text
INT8 Dequant-GEMV benchmark 数据
INT8 Dequant-GEMV 优化文档
```

### Day 7：文档包装

任务：

1. 整理 README；
2. 汇总 benchmark；
3. 写迁移思考；
4. 写简历描述；
5. 写报名材料描述；
6. 检查 GitHub 仓库。

产出：

```text
完整开源项目展示页
简历项目描述
夏令营报名材料项目说明
```

---

## 21. 项目验收标准

项目完成后，应至少满足以下标准：

### 21.1 代码标准

1. 仓库结构清晰；
2. 每个算子独立目录；
3. 每个 CUDA kernel 有基本注释；
4. Python 调用接口清晰；
5. 测试脚本可运行；
6. Benchmark 脚本可运行；
7. 没有大体积无关文件；
8. Git commit 记录清楚。

### 21.2 功能标准

至少完成：

1. RMSNorm PyTorch reference；
2. RMSNorm CUDA naive；
3. RMSNorm CUDA optimized；
4. RoPE PyTorch reference；
5. RoPE CUDA kernel；
6. INT8 Dequant-GEMV PyTorch reference；
7. INT8 Dequant-GEMV fused CUDA kernel；
8. 三类算子的 correctness test；
9. 三类算子的 benchmark。

### 21.3 文档标准

至少完成：

1. README.md；
2. project_design.md；
3. RMSNorm 文档；
4. RoPE 文档；
5. INT8 Dequant-GEMV 文档；
6. Benchmark 报告；
7. CUDA 到 TileLang/MXMACA 迁移思考；
8. 简历项目描述。

### 21.4 展示标准

GitHub 首页应能让评审快速看到：

1. 你为什么做这个项目；
2. 你实现了哪些算子；
3. 你怎么验证正确性；
4. 你怎么测试性能；
5. 你做了哪些优化；
6. 你获得了什么结果；
7. 你后续如何迁移到国产 GPU 软件栈。

---

## 22. 风险与应对方案

### 22.1 风险一：Jetson 环境配置耗时

问题：

Jetson 上 PyTorch、CUDA、Python 包版本可能存在兼容问题。

应对：

1. 第一天优先解决环境；
2. 不要一开始写复杂 kernel；
3. 先跑通最小 CUDA extension；
4. 记录环境配置文档；
5. 遇到 PyTorch extension 问题时，可以先用 nvcc 编译独立 CUDA 程序。

### 22.2 风险二：CUDA kernel 调试困难

问题：

新手容易遇到非法内存访问、结果错误、维度映射错误。

应对：

1. 先用小 shape；
2. 先写 CPU/PyTorch reference；
3. 每一步都做 correctness test；
4. 使用简单清晰的 index 逻辑；
5. 先保证正确，再优化性能。

### 22.3 风险三：优化效果不明显

问题：

自定义 CUDA kernel 未必一定快于 PyTorch。

应对：

1. 不虚构性能数据；
2. 重点写清楚分析过程；
3. 对比 naive 和 optimized；
4. 分析为什么某些规模下没有优势；
5. 强调学习到的性能瓶颈；
6. 对小算子关注 kernel launch 和访存开销。

### 22.4 风险四：项目做得太散

问题：

同时做太多算子容易每个都不深入。

应对：

1. RMSNorm 必须做扎实；
2. RoPE 做完整正确性和 benchmark；
3. INT8 Dequant-GEMV 作为亮点；
4. FlashAttention 暂时不做；
5. 每个算子都要有文档闭环。

### 22.5 风险五：与夏令营方向关联不够明显

问题：

Jetson 是 NVIDIA 平台，不是国产 GPU 平台。

应对：

1. README 明确说明 CUDA 是算子训练平台；
2. 文档中写清楚迁移到 TileLang/MXMACA 的计划；
3. 强调 GPU 算子开发共性能力；
4. 把项目重点放在 kernel、benchmark、优化，而不是 Jetson 应用；
5. 报名材料中主动说明后续希望参与国产 GPU 算子开发。

---

## 23. README 首页建议结构

README 建议如下组织：

```text
# EdgeLLM-KernelBench

## Introduction
一句话说明项目。

## Why this project
说明端侧大模型推理为什么需要算子优化。

## Hardware
说明 Jetson Orin Nano Super 8GB。

## Operators
列出 RMSNorm、RoPE、INT8 Dequant-GEMV。

## Project Structure
展示目录结构。

## Quick Start
说明安装、编译、测试。

## Correctness
展示误差验证方法。

## Benchmark
展示 benchmark 命令和结果表格。

## Optimization Notes
总结 shared memory、warp reduction、fusion 等方法。

## CUDA to TileLang/MXMACA
说明迁移思考。

## Roadmap
说明后续计划。
```

---

## 24. 简历项目描述

可以在简历中写成：

```text
EdgeLLM-KernelBench：面向端侧大模型推理的 GPU 算子优化实践平台

基于 NVIDIA Jetson Orin Nano Super 8GB 构建端侧 GPU 算子优化实践项目，围绕大模型推理中的 RMSNorm、RoPE、INT8 Dequant-GEMV 等核心算子，完成 PyTorch baseline、CUDA C++ kernel 实现、正确性验证、自动化 Benchmark 与性能分析。项目重点分析小 batch LLM 推理场景下的访存瓶颈、规约开销、kernel launch overhead 与算子融合收益，并通过 shared memory、warp-level reduction、连续访存和 fused dequant-gemv 等方法进行优化。项目同时整理 CUDA 算子到 TileLang/MXMACA 国产 GPU 软件栈的迁移思路，为后续参与国产 GPU 大模型推理算子开发与优化任务做准备。
```

---

## 25. 夏令营报名材料项目描述

报名材料中可以写成：

```text
我近期主要在做端侧 AI 推理部署与服务化项目，已经完成 RK3588 平台上视觉模型和大语言模型的部署、模型注册、OpenAI-like API 服务、worker 后端、metrics 指标和板端验证。通过这一项目，我对端侧推理系统的工程链路有了较完整的理解，但也意识到真正影响大模型推理效率的核心往往在底层算子、访存模式、并行策略和硬件软件栈协同优化。

因此，我进一步计划基于 Jetson Orin Nano Super 8GB 搭建一个面向大模型推理的 GPU 算子优化实践项目，优先实现 RMSNorm、RoPE、INT8 Dequant-GEMV 等典型算子，完成从 PyTorch baseline、CUDA kernel 编写、正确性验证、Benchmark 到优化分析的完整闭环。虽然当前平台使用 CUDA 生态，但项目重点训练的是 GPU 算子开发的共性能力，包括线程块划分、访存合并、shared memory、warp-level reduction、算子融合和性能 profiling。后续希望将这些经验迁移到 TileLang/MXMACA 等国产 GPU 软件栈中，参与真实的大模型推理关键算子优化任务。
```

---

## 26. 后续扩展方向

如果前三个算子完成较好，后续可以扩展：

1. SiLU / GELU activation kernel；
2. SwiGLU fused kernel；
3. LayerNorm；
4. FP16 GEMV；
5. INT4 Dequant-GEMV；
6. Batch GEMV；
7. 简化版 Attention；
8. 简化版 FlashAttention；
9. TileLang RMSNorm；
10. TileLang Dequant-GEMV；
11. 与 TensorRT plugin 的对比；
12. 与 RK3588 推理服务项目联动，形成端侧异构推理系统展示。

---

## 27. 最终项目成果形式

最终项目应包含以下成果：

```text
1. 一个 GitHub 开源仓库；
2. 三类可运行 CUDA 算子；
3. 三类 PyTorch reference；
4. 三类 correctness test；
5. 三类 benchmark 脚本；
6. CSV 性能数据；
7. Markdown 性能报告；
8. 算子优化文档；
9. CUDA 到 TileLang/MXMACA 迁移思考；
10. 简历项目描述；
11. 夏令营报名材料项目描述。
```

---

## 28. 项目亮点总结

本项目的亮点可以总结为：

1. 方向贴近大模型推理底层优化；
2. 不是简单模型部署，而是 CUDA 算子开发；
3. 选择 RMSNorm、RoPE、INT8 Dequant-GEMV，难度递进合理；
4. 覆盖 reference、kernel、test、benchmark、report 完整闭环；
5. 与已有 RK3588 端侧推理项目形成互补；
6. 具备向 TileLang/MXMACA 国产 GPU 软件栈迁移的延展性；
7. 适合写入简历和夏令营报名材料；
8. 可以在较短时间内完成一个真实、专业、可展示的开源项目。

---

## 29. 项目最终定位

EdgeLLM-KernelBench 最终可以定位为：

> 一个基于 Jetson/CUDA 的端侧大模型推理算子优化实践平台，用于系统性训练 GPU kernel 编写、正确性验证、Benchmark 测试和性能优化能力，并为后续参与国产 GPU 大模型推理关键算子开发打基础。

它与已有 RK3588 项目共同构成如下能力闭环：

```text
模型部署能力
    +
推理服务工程能力
    +
GPU 算子优化能力
    +
国产 GPU 软件栈迁移潜力
```

这比单纯写“跑过模型”“会部署 YOLO”“会调用大模型 API”更有辨识度，也更符合 GPU 算子开发类夏令营对候选人的期待。