

| 算子 | 用途 | 优化重点 |
|---|---|---|
| RMSNorm | Transformer 归一化 | Warp Reduce、float4、FP16、half2 |
| RoPE | 旋转位置编码 | 向量化、sin/cos 访问、Q/K 融合 |
| INT8 Dequant-GEMV | 量化线性层 | 反量化融合、INT8 读取、FP32 累加 |

当前非目标：

- 完整训练系统；
- 多机多卡；
- 重写通用推理框架；
- 一次性支持全部 dtype；
- 为追求“看起来更快”而选择性报告数据。

---

## 3. 硬件与软件环境

### 3.1 硬件

```text
设备：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
内存：8GB
GPU：Ampere
Compute Capability：8.7
编译目标：sm_87
Active GPU TPCs：4
```

正式 Benchmark 已使用：

```text
NV Power Mode：MAXN_SUPER
CPU：1.728 GHz
GPU：1.020 GHz
EMC：3.199 GHz
```

正式测量前执行：

```bash
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show
```

### 3.2 软件

```text
Ubuntu 22.04 LTS
JetPack 6.2.1+b38
L4T R36.4.7
Kernel 5.15.148-tegra
CUDA 12.6
nvcc 12.6.68
Python 3.10.12
PyTorch 2.8.0 Jetson ARM64 CUDA 版本
NumPy 1.26.1
pytest 9.1.1
g++ 11.4.0
CMake 3.22.1
Ninja 1.10.1
Git 2.34.1
```

虚拟环境：

```bash
source ~/venvs/edge-llm-kernelbench/bin/activate
```

推荐变量：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.7
```

Jetson 内存有限，扩展编译统一使用：

```bash
MAX_JOBS=2
```

---

## 4. 仓库与目录

项目目录：

```text
/home/liujiayu/edge-llm-kernelbench
```

远端：

```text
git@github.com:mystery5820/edge-llm-kernelbench.git
```

推荐结构：

```text
edge-llm-kernelbench/
├── README.md
├── .gitignore
├── docs/
│   ├── project_design.md
│   ├── project_design_codex.md
│   ├── current_progress_codex_handoff.md
│   └── 各算子优化记录
├── python/edge_kernelbench/
│   ├── __init__.py
│   ├── rmsnorm.py
│   ├── rmsnorm_cuda.py
│   ├── rope.py
│   ├── rope_cuda.py
│   └── int8_dequant_gemv*.py
├── kernels/
│   ├── rmsnorm/
│   │   ├── rmsnorm.cpp
│   │   ├── rmsnorm_kernel.cu
│   │   ├── rmsnorm_warp_kernel.cu
│   │   └── rmsnorm_float4_kernel.cu
│   ├── rope/
│   └── int8_dequant_gemv/
├── tests/
├── benchmarks/
├── results/
├── scripts/
└── build/                 # 不提交 Git
```

---

## 5. 软件架构

### 5.1 Reference 层

职责：

- 表达数学定义；
- 作为 CUDA 正确性基准；
- 支持 CPU/CUDA；
- 支持梯度和基础 dtype；
- 提供清晰异常信息。

RMSNorm API：

```python
rmsnorm_reference(x, weight, eps=1e-6)
RMSNormReference(hidden_size, eps=1e-6)
```

### 5.2 Python CUDA 封装层

文件：

```text
python/edge_kernelbench/rmsnorm_cuda.py
```

职责：

- 查找项目根目录和源文件；
- 使用 `torch.utils.cpp_extension.load()`；
- 设置 C++/CUDA 编译参数；
- 缓存模块；
- 提供多实现 Python API。

现有 API：

```python
load_rmsnorm_cuda_extension()
rmsnorm_cuda()
rmsnorm_cuda_warp()
```

计划新增：

```python
rmsnorm_cuda_float4()
rmsnorm_cuda_fp16()
```

### 5.3 C++ / PyBind11 层

文件：

```text
kernels/rmsnorm/rmsnorm.cpp
```

职责：

- 统一输入检查；
- 声明各 CUDA Launcher；
- 调用对应实现；
- 注册 Python API。

当前接口：

```text
forward
forward_warp
```

计划：

```text
forward_float4
forward_fp16
```

### 5.4 CUDA 层

每个优化版本使用独立 `.cu` 文件。原则：

- 不覆盖稳定旧版本；
- 每个版本有独立 Launcher；
- 每个版本可被 Benchmark 同时调用；
- 每个版本有独立测试；
- 无收益版本也保留。

---

## 6. RMSNorm 设计

### 6.1 数学定义

```text
mean_square = Σ(xi²) / hidden_size
inverse_rms = 1 / sqrt(mean_square + eps)
yi = xi × inverse_rms × weighti
```

沿最后一维归一化。任意前置维度展平为：

```text
[rows, hidden_size]
rows = x.numel() / hidden_size
```

### 6.2 Naive CUDA

设计：

- 一个 Block 处理一行；
- 256 线程；
- 每线程局部平方和；
- 每线程写 Shared Memory；
- 二叉树规约；
- 多轮 `__syncthreads()`；
- 第二次遍历写输出。

优势：直观、可靠、易验证。  
不足：Shared Memory 访问和同步较多。

### 6.3 Warp Shuffle CUDA

设计：

- 256 线程，8 个 Warp；
- Warp 内 `__shfl_down_sync()`；
- 每个 Warp 仅写一个 Shared Memory 值；
- 第一个 Warp 汇总；
- Shared Memory 从 256 个 float 降到 8 个 float。

已有实验结论：

```text
数值正确
Warp 相对 Naive 约 0.983x～1.002x
没有形成稳定加速
```

这说明减少 Shared Memory 和规约同步并不必然改善端到端延迟。可能瓶颈包括 Kernel launch、全局内存和额外指令，需 Profiling 验证。

### 6.4 float4 CUDA

Phase 3 目标：

- 在 Warp Shuffle 基础上优化全局访存；
- 一次处理 4 个 FP32 元素；
- 向量化输入、权重和输出；
- 保留尾部和非对齐安全路径。

必须满足：

1. 16 字节对齐时使用 `float4`。
2. `hidden_size / 4` 主体向量化。
3. 余数使用标量。
4. 行首地址未对齐时整行退回标量。
5. 独立 Launcher：`rmsnorm_float4_cuda_launcher`。
6. C++ 接口：`forward_float4`。
7. Python API：`rmsnorm_cuda_float4`。
8. 对齐、未对齐、尾部、空输入都要测试。
9. Benchmark 同时比较四种实现。

### 6.5 后续 RMSNorm 优化

- FP16 输入、FP32 累加；
- `half2`；
- 一个 Warp 一行；
- 一个 Block 多行；
- 不同 hidden size 自动选择线程数；
- 模板化 dtype；
- CUDA Graph；
- 静态扩展构建；
- 参数自动调优。

---

## 7. RoPE 设计

公式：

```text
y_even = x_even × cosθ - x_odd × sinθ
y_odd  = x_even × sinθ + x_odd × cosθ
```

建议输入：

```text
[batch, sequence, heads, head_dim]
```

要求：

- `head_dim` 为偶数；
- 明确 interleaved 或 split-half 布局；
- position 和 sin/cos 表匹配。

优化方向：

- 一线程处理一对通道；
- `float2` / `half2`；
- 合并 sin/cos 读取；
- 优化索引和地址计算；
- Q/K 融合；
- 预计算表与运行时计算对比。

测试：

- 零角度；
- 固定角度；
- 多 batch、sequence、head；
- 奇数 head_dim 拒绝；
- FP32/FP16 容差；
- 输出元数据。

---

## 8. INT8 Dequant-GEMV 设计

首版量化：

```text
per-channel symmetric INT8
W_fp = W_int8 × scale
```

目标：

```text
y = dequant(W_int8) × x
```

要求：

- 不生成完整 FP32 中间权重；
- 反量化与 GEMV 融合；
- INT8 权重连续读取；
- FP32 累加；
- scale 在输出前融合；
- 后续尝试 `char4` / 打包读取。

比较：

1. PyTorch 显式反量化 + matmul；
2. 自定义融合 Dequant-GEMV；
3. 可选其他库基准；
4. 多种 M、K 和 batch。

---

## 9. 自动化测试

每个实现至少覆盖：

- Reference 正确性；
- CUDA 与 Reference；
- CUDA 版本交叉对比；
- 多形状；
- 边界输入；
- 异常输入；
- 输出 shape/dtype/device；
- 全量回归。

FP32 建议容差：

```python
rtol=1e-5
atol=1e-6
```

RMSNorm 形状：

```text
(1024,)
(4, 128)
(2, 3, 257)
(8, 1024)
(1, 4096)
(16, 4096)
(128, 4096)
```

float4 额外覆盖：

```text
hidden_size % 4 == 0
hidden_size % 4 != 0
16 字节对齐
非对齐 storage_offset
空行
全零
常量输入
```

全量测试：

```bash
MAX_JOBS=2 PYTHONPATH=python \
python -m pytest -v
```

---

## 10. Benchmark 规范

原则：

- 使用 `torch.cuda.Event`；
- 预热后测量；
- 每轮重复调用；
- 使用中位数作为主要指标；
- 同时记录 mean、median、min、P95；
- 每组先做正确性检查；
- 正式测试锁频；
- 避免其他 GPU 负载；
- 保存 CSV 和终端日志。

正式参数：

```text
warmup=20
rounds=30
repeats=50
```

当前输入：

```text
rows=1, hidden=1024
rows=1, hidden=4096
rows=16, hidden=4096
rows=128, hidden=4096
```

运行：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)

MAX_JOBS=2 PYTHONPATH=python \
python benchmarks/benchmark_rmsnorm.py \
  --warmup 20 \
  --rounds 30 \
  --repeats 50 \
  | tee "results/rmsnorm_comparison_console_${STAMP}.log"
```

CSV 必须指定：

```python
lineterminator="\n"
```

---

## 11. Profiling 规范

Nsight Systems：

- Python/C++/CUDA 调用链；
- Kernel launch 间隔；
- 隐式同步；
- Reference 的多 Kernel 结构；
- 启动开销占比。

Nsight Compute：

- Kernel Duration；
- DRAM / Memory Throughput；
- Global Load/Store；
- Shared Memory；
- Warp Stall；
- Occupancy；
- Registers；
- Branch Efficiency；
- L1/L2；
- Eligible Warps。

Warp 无收益时重点分析：

- Shared Memory 是否根本不是瓶颈；
- 全局内存是否主导；
- 小规模场景是否 launch-bound；
- Warp 版本是否增加指令；
- 两次 Block 同步的影响；
- 输出阶段是否主导。

---

## 12. 工程规范

Codex 和人工修改都必须遵守：

1. 修改前 `git status --short`。
2. 先读文件，不凭假设覆盖。
3. 不删除 Naive 和 Warp 稳定实现。
4. 新优化使用独立 API。
5. 先语法检查，再专项测试，再全量测试。
6. Benchmark 前锁频。
7. 正式结果保存 CSV 和日志。
8. `git diff --check` 必须通过。
9. 不提交 `build/`、`.so`、`.o`、缓存。
10. 每阶段单独提交和打标签。
11. 代码关键逻辑使用详细中文注释。
12. 性能结论必须基于正式数据。

建议提交：

```text
add RMSNorm float4 vectorized kernel and tests
add RMSNorm float4 benchmark comparison
document RMSNorm float4 optimization results
```

阶段标签：

```text
phase1-rmsnorm-naive-baseline
phase2-rmsnorm-warp-comparison
phase3-rmsnorm-float4-comparison
phase4-rmsnorm-fp16-comparison
```

---

## 13. 路线图

### Phase 0
环境、仓库、CUDA Hello、PyTorch Extension。

### Phase 1
RMSNorm Reference、Naive CUDA、测试、基线、标签。

### Phase 2
Warp Shuffle、测试、三方 Benchmark、无稳定加速结论、标签。

### Phase 3
float4、对齐与尾部测试、四方 Benchmark、Profiling、标签。

### Phase 4
FP16、FP32 累加、half2、精度与性能比较。

### Phase 5
RMSNorm Nsight 分析和技术总结。

### Phase 6
RoPE Reference、CUDA、向量化、FP16、Benchmark。

### Phase 7
INT8 Dequant-GEMV、融合反量化、向量化、Benchmark。

### Phase 8
一键脚本、结果可视化、README、项目总结和开源展示。

---

## 14. 阶段验收标准

每个 CUDA 版本完成时必须满足：

- 编译成功；
- Python API 可调用；
- Reference 对比通过；
- 多形状和边界测试通过；
- 全量测试通过；
- 正式 Benchmark 完成；
- CSV 和日志保存；
- 结论如实记录；
- `git diff --check` 通过；
- 推送 GitHub；
- 工作区 clean；
- 创建阶段标签。

---

## 15. Codex 接手入口

Codex 首先阅读：

```text
docs/project_design_codex.md
docs/current_progress_codex_handoff.md
README.md
kernels/rmsnorm/rmsnorm.cpp
python/edge_kernelbench/rmsnorm_cuda.py
benchmarks/benchmark_rmsnorm.py
tests/test_rmsnorm*.py
```

第一条命令：

```bash
git status --short
```

每轮修改后至少运行：

```bash
python -m py_compile <修改的 Python 文件>

MAX_JOBS=2 PYTHONPATH=python \
python -m pytest -v

git diff --check
git status --short
```

本项目最终价值不是某一个“最快 Kernel”，而是完整展示：

```text
算子理解
→ Reference
→ CUDA
→ PyTorch 接入
→ 测试
→ Benchmark
→ Profiling
→ 结果解释
→ 工程管理
```