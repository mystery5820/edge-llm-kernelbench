# EdgeLLM-KernelBench 当前进度与 Codex 交接文档

> 建议工程路径：`docs/current_progress_codex_handoff.md`  
> 交接日期：2026-07-12  
> 当前项目：`edge-llm-kernelbench`  
> 当前主线：RMSNorm CUDA 算子优化  
> 下一任务：Phase 3 `float4` 向量化访存优化

---

## 1. 仓库信息

项目目录：

```text
/home/liujiayu/edge-llm-kernelbench
```

远端：

```text
git@github.com:mystery5820/edge-llm-kernelbench.git
```

主分支：

```text
main
```

最近已验证提交：

```text
6bc9be7 add RMSNorm warp shuffle kernel and benchmark comparison
7618c25 add RMSNorm baseline benchmark and results
c0cff4e add RMSNorm naive CUDA kernel and tests
2718367 add RMSNorm PyTorch reference and tests
f980235 add detailed project design document
8ddee34 initialize project structure and environment documentation
```

已有标签：

```text
phase1-rmsnorm-naive-baseline
phase2-rmsnorm-warp-comparison
```

开始 Phase 3 之前最后一次已验证状态：

```text
HEAD = 6bc9be7
origin/main 已同步
working tree clean
```

随后用户已经开始创建 Phase 3 文件，因此 Codex 必须先检查当前实际工作区。

---

## 2. 当前环境

硬件：

```text
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
内存：8GB
GPU：Ampere
Compute Capability：8.7
目标架构：sm_87
```

软件：

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
```

虚拟环境：

```bash
source ~/venvs/edge-llm-kernelbench/bin/activate
```

编译设置：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.7
MAX_JOBS=2
```

正式 Benchmark 模式：

```text
NV Power Mode：MAXN_SUPER
CPU：1.728 GHz
GPU：1.020 GHz
EMC：3.199 GHz
```

---

## 3. 已完成内容

### 3.1 工程骨架

已完成：

- Git/GitHub；
- README；
- 项目设计文档；
- `python/`、`kernels/`、`tests/`、`benchmarks/`、`results/`；
- `.gitignore`；
- CUDA Hello Kernel；
- PyTorch CUDA Extension 验证；
- `sm_87` 编译验证。

### 3.2 RMSNorm PyTorch Reference

文件：

```text
python/edge_kernelbench/rmsnorm.py
```

接口：

```python
rmsnorm_reference(x, weight, eps=1e-6)
RMSNormReference(hidden_size, eps=1e-6)
```

功能：

- CPU/CUDA；
- 输入和 weight 检查；
- hidden size 检查；
- device 检查；
- eps 检查；
- FP16/BF16 内部 FP32 计算；
- 输出恢复输入 dtype；
- 支持反向传播。

测试：

```text
tests/test_rmsnorm.py
19 项
```

### 3.3 RMSNorm Naive CUDA

文件：

```text
kernels/rmsnorm/rmsnorm.cpp
kernels/rmsnorm/rmsnorm_kernel.cu
python/edge_kernelbench/rmsnorm_cuda.py
tests/test_rmsnorm_cuda.py
```

API：

```python
rmsnorm_cuda(x, weight, eps=1e-6)
```

实现：

- FP32；
- 一个 Block 一行；
- 256 线程；
- Shared Memory 二叉树规约；
- 当前 PyTorch CUDA Stream；
- CUDAGuard；
- Kernel launch check；
- 空行支持；
- 统一异常检查。

### 3.4 RMSNorm Warp Shuffle

文件：

```text
kernels/rmsnorm/rmsnorm_warp_kernel.cu
tests/test_rmsnorm_warp.py
```

API：

```python
rmsnorm_cuda_warp(x, weight, eps=1e-6)
```

PyBind：

```text
forward_warp
```

实现：

- FP32；
- 256 线程，8 个 Warp；
- `__shfl_down_sync()`；
- 每 Warp 一个 Shared Memory 结果；
- 第一个 Warp 汇总；
- 与 Naive 共用 C++ 输入检查。

最近全量测试：

```text
49 passed in 3.37s
```

---

## 4. 当前关键文件和 API

### 4.1 `kernels/rmsnorm/rmsnorm.cpp`

当前应包含：

```text
rmsnorm_cuda_launcher
rmsnorm_warp_cuda_launcher
validate_rmsnorm_inputs
rmsnorm_forward
rmsnorm_warp_forward
PYBIND11_MODULE
```

当前注册：

```text
forward
forward_warp
```

Phase 3 需要新增：

```text
rmsnorm_float4_cuda_launcher
rmsnorm_float4_forward
forward_float4
```

### 4.2 `python/edge_kernelbench/rmsnorm_cuda.py`

当前 sources 应包含：

```text
rmsnorm.cpp
rmsnorm_kernel.cu
rmsnorm_warp_kernel.cu
```

当前 API：

```python
load_rmsnorm_cuda_extension()
rmsnorm_cuda()
rmsnorm_cuda_warp()
```

Phase 3 需要：

```text
加入 rmsnorm_float4_kernel.cu
新增 rmsnorm_cuda_float4()
```

### 4.3 `benchmarks/benchmark_rmsnorm.py`

当前比较：

```text
pytorch_reference
cuda_naive
cuda_warp
```

当前功能：

- CUDA Event；
- warmup / rounds / repeats；
- mean / median / min / P95；
- speedup_vs_reference；
- Naive 和 Warp 正确性检查；
- CSV 和控制台日志；
- CSV 已设置 `lineterminator="\n"`。

Phase 3 需要加入：

```text
cuda_float4
Float4 correctness maximum error
Float4 vs Reference
Float4 vs Naive
Float4 vs Warp
```

---

## 5. 已保存 Benchmark 结果

### 5.1 Naive 基线

文件：

```text
results/rmsnorm_baseline_20260712_161636.csv
results/rmsnorm_baseline_console_20260712_161631.log
```

正式参数：

```text
warmup=20
rounds=30
repeats=50
```

Naive 相对 PyTorch Reference：

```text
rows=1,   hidden=1024：7.807x
rows=1,   hidden=4096：8.055x
rows=16,  hidden=4096：7.918x
rows=128, hidden=4096：4.237x
```

### 5.2 Warp 对比

文件：

```text
results/rmsnorm_warp_comparison_20260712_175730.csv
results/rmsnorm_warp_comparison_console_20260712_175725.log
```

最终数据：

```text
rows=1, hidden=1024
Naive median：0.029276 ms
Warp median： 0.029414 ms
Warp vs Naive：0.995x

rows=1, hidden=4096
Naive median：0.028436 ms
Warp median： 0.028700 ms
Warp vs Naive：0.991x

rows=16, hidden=4096
Naive median：0.028725 ms
Warp median： 0.029208 ms
Warp vs Naive：0.983x

rows=128, hidden=4096
Naive median：0.055555 ms
Warp median： 0.055449 ms
Warp vs Naive：1.002x
```

正式结论：

```text
Warp Shuffle 数值正确，但没有形成稳定加速。
```

必须保留该结论，不得为了宣传效果改写。

---

## 6. 当前 Phase 3 的已知问题

用户已经创建：

```text
kernels/rmsnorm/rmsnorm_float4_kernel.cu
```

但检查结果显示，该文件实际误粘贴成 Warp Kernel 的重复内容。

检查 float4 关键符号时，只在注释中找到一次 `float4`：

```bash
grep -nE \
  "rmsnorm_float4_kernel|rmsnorm_float4_cuda_launcher|is_float4_aligned|float4" \
  kernels/rmsnorm/rmsnorm_float4_kernel.cu
```

检查 Warp 符号时找到：

```text
warp_reduce_sum
rmsnorm_warp_kernel
shared_warp_sums
rmsnorm_warp_cuda_launcher
```

因此该文件目前很可能不正确。

最后建议用户执行过以下命令，但尚未收到执行结果：

```bash
rm kernels/rmsnorm/rmsnorm_float4_kernel.cu
touch kernels/rmsnorm/rmsnorm_float4_kernel.cu
wc -l kernels/rmsnorm/rmsnorm_float4_kernel.cu
```

Codex 不能假设文件已经清空，必须先检查。

---

## 7. Codex 第一轮操作

### Step 1：检查状态

```bash
cd ~/edge-llm-kernelbench
source ~/venvs/edge-llm-kernelbench/bin/activate

git status --short
git log -3 --oneline
git tag --list --sort=-creatordate | head
```

### Step 2：检查 float4 文件

```bash
wc -l kernels/rmsnorm/rmsnorm_float4_kernel.cu

grep -nE \
  "rmsnorm_float4_kernel|rmsnorm_float4_cuda_launcher|is_float4_aligned|make_float4|rmsnorm_warp_kernel|rmsnorm_warp_cuda_launcher" \
  kernels/rmsnorm/rmsnorm_float4_kernel.cu \
  | head -n 50
```

如果仍为重复 Warp 文件：

```bash
rm kernels/rmsnorm/rmsnorm_float4_kernel.cu
touch kernels/rmsnorm/rmsnorm_float4_kernel.cu
```

---

## 8. Phase 3 实现任务

### 8.1 正确实现 float4 Kernel

目标文件：

```text
kernels/rmsnorm/rmsnorm_float4_kernel.cu
```

必须包含：

```cpp
__device__ __forceinline__ bool is_float4_aligned(...)
__device__ __forceinline__ float warp_reduce_sum_float4(...)
__global__ void rmsnorm_float4_kernel(...)
torch::Tensor rmsnorm_float4_cuda_launcher(...)
```

要求：

- FP32；
- 一个 Block 处理一行；
- 256 线程；
- Warp Shuffle 两级规约；
- `float4` 输入读取；
- `float4` 权重读取；
- `float4` 输出写回；
- `hidden_size % 4` 尾部标量处理；
- 行首未对齐时标量回退；
- CUDAGuard；
- PyTorch 当前 Stream；
- Kernel launch check；
- 空行支持。

### 8.2 修改 C++ 接口

文件：

```text
kernels/rmsnorm/rmsnorm.cpp
```

新增声明：

```cpp
torch::Tensor rmsnorm_float4_cuda_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
);
```

新增：

```cpp
torch::Tensor rmsnorm_float4_forward(...)
```

复用：

```cpp
validate_rmsnorm_inputs(...)
```

注册：

```cpp
module.def(
    "forward_float4",
    &rmsnorm_float4_forward,
    "RMSNorm float4 vectorized CUDA forward"
);
```

### 8.3 修改 Python 加载器

文件：

```text
python/edge_kernelbench/rmsnorm_cuda.py
```

将：

```text
rmsnorm_float4_kernel.cu
```

加入 sources。

新增：

```python
def rmsnorm_cuda_float4(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    extension = load_rmsnorm_cuda_extension()
    return extension.forward_float4(
        x,
        weight,
        float(eps),
    )
```

### 8.4 编译检查

```bash
MAX_JOBS=2 PYTHONPATH=python python -c "
from edge_kernelbench.rmsnorm_cuda import load_rmsnorm_cuda_extension

module = load_rmsnorm_cuda_extension(verbose=True)

print('forward:', hasattr(module, 'forward'))
print('forward_warp:', hasattr(module, 'forward_warp'))
print('forward_float4:', hasattr(module, 'forward_float4'))

assert hasattr(module, 'forward')
assert hasattr(module, 'forward_warp')
assert hasattr(module, 'forward_float4')
"
```

---

## 9. Phase 3 测试要求

新文件：

```text
tests/test_rmsnorm_float4.py
```

至少覆盖：

1. 固定值；
2. `(1024,)`；
3. `(4, 128)`；
4. `(2, 3, 257)`；
5. `(1, 4096)`；
6. `(16, 4096)`；
7. `(128, 4096)`；
8. `hidden_size=1025`；
9. `hidden_size=1026`；
10. `hidden_size=1027`；
11. 全零；
12. 常量；
13. 空行；
14. 输出元数据；
15. float4 vs Reference；
16. float4 vs Naive；
17. float4 vs Warp；
18. 非对齐回退。

非对齐、但仍连续的输入可这样构造：

```python
base = torch.randn(
    rows * hidden_size + 1,
    device="cuda",
    dtype=torch.float32,
)

x = base[1:].view(
    rows,
    hidden_size,
)

assert x.is_contiguous()
assert x.storage_offset() == 1
```

这样首地址偏移 4 字节，可触发非 16 字节对齐路径。

专项测试：

```bash
MAX_JOBS=2 PYTHONPATH=python \
python -m pytest tests/test_rmsnorm_float4.py -v
```

全量测试：

```bash
MAX_JOBS=2 PYTHONPATH=python \
python -m pytest -v
```

已有 49 项必须继续通过。

---

## 10. Phase 3 Benchmark 要求

修改：

```text
benchmarks/benchmark_rmsnorm.py
```

新增实现：

```text
cuda_float4
```

每组输出：

```text
Naive correctness maximum error
Warp correctness maximum error
Float4 correctness maximum error
```

测量：

```text
pytorch_reference
cuda_naive
cuda_warp
cuda_float4
```

计算：

```text
float4 vs reference
float4 vs naive
float4 vs warp
```

正式前：

```bash
sudo nvpmodel -q
sudo jetson_clocks
sudo jetson_clocks --show
```

正式运行：

```bash
STAMP=$(date +%Y%m%d_%H%M%S)

MAX_JOBS=2 PYTHONPATH=python \
python benchmarks/benchmark_rmsnorm.py \
  --warmup 20 \
  --rounds 30 \
  --repeats 50 \
  | tee "results/rmsnorm_float4_comparison_console_${STAMP}.log"
```

删除冒烟结果，只保留最终 CSV 和日志。

---

## 11. 提交前检查

```bash
python -m py_compile \
  python/edge_kernelbench/rmsnorm_cuda.py \
  tests/test_rmsnorm_float4.py \
  benchmarks/benchmark_rmsnorm.py

MAX_JOBS=2 PYTHONPATH=python \
python -m pytest -v

git diff --check
git status --short
```

若 CSV 出现 `^M`：

```bash
sed -i 's/\r$//' <csv-file>
```

确认 Benchmark 使用：

```python
csv.DictWriter(
    csv_file,
    fieldnames=fieldnames,
    lineterminator="\n",
)
```

不要提交：

```text
build/
*.so
*.o
.pytest_cache/
__pycache__/
临时补丁脚本
冒烟 CSV
备份文件
```

---

## 12. 建议提交与标签

提交：

```bash
git add \
  kernels/rmsnorm/rmsnorm_float4_kernel.cu \
  kernels/rmsnorm/rmsnorm.cpp \
  python/edge_kernelbench/rmsnorm_cuda.py \
  tests/test_rmsnorm_float4.py \
  benchmarks/benchmark_rmsnorm.py \
  results/<正式CSV> \
  results/<正式日志>

git commit -m "add RMSNorm float4 vectorized kernel and benchmark comparison"
git push
```

标签：

```bash
git tag -a phase3-rmsnorm-float4-comparison \
  -m "RMSNorm float4 vectorized kernel, tests and benchmark comparison"

git push origin phase3-rmsnorm-float4-comparison
```

---

## 13. Codex 修改约束

Codex 必须遵守：

- 不覆盖 Naive 和 Warp；
- 不修改已有标签；
- 不改写历史结果；
- 不提交 build 产物；
- 编译使用 `MAX_JOBS=2`；
- 架构为 `sm_87`；
- 修改前读取实际内容；
- 长文件修改后检查关键符号；
- 测试失败不能通过删除测试解决；
- 性能结论必须来自正式 Benchmark；
- 无收益优化也要保留；
- 核心 CUDA 逻辑使用详细中文注释；
- 未经用户明确要求，不自动 commit。

---

## 14. 可直接交给 Codex 的提示词

```text
请先阅读 docs/project_design_codex.md 和
docs/current_progress_codex_handoff.md。

不要立即修改代码。先执行 git status --short，并检查
kernels/rmsnorm/rmsnorm_float4_kernel.cu 的实际内容。

当前稳定提交为 6bc9be7，已经完成 RMSNorm PyTorch Reference、
Naive CUDA、Warp Shuffle CUDA，49 项测试通过。
Phase 3 的 float4 文件可能仍误包含 Warp Kernel 的重复代码。

请保留 Naive 和 Warp 版本，完成：
1. 正确 float4 CUDA Kernel；
2. C++ forward_float4；
3. Python rmsnorm_cuda_float4；
4. float4 自动化测试；
5. 四方 Benchmark；
6. 全量回归；
7. 正式 CSV 和日志。

编译使用 MAX_JOBS=2，目标 sm_87。
修改后展示 diff、测试和 Benchmark 结果。
不要自行提交，除非我明确要求。
```