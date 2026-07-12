# EdgeLLM-KernelBench

面向端侧大模型推理的 GPU 算子优化实践平台。

本项目基于 NVIDIA Jetson Orin Nano Super 8GB，围绕大模型推理中的典型 GPU 算子，完成 PyTorch 参考实现、CUDA Kernel 开发、正确性验证、性能测试与优化分析。

## 开发环境

- Hardware: NVIDIA Jetson Orin Nano Super 8GB
- JetPack: 6.2.1
- L4T: R36.4.7
- Ubuntu: 22.04 LTS
- CUDA Toolkit: 12.6
- GPU Compute Capability: 8.7（sm_87）
- Python: 3.10.12
- PyTorch: 2.8.0
- GCC/G++: 11.4.0
- CMake: 3.22.1
- Ninja: 1.10.1

## 计划实现的算子

1. RMSNorm
2. RoPE
3. INT8 Dequant-GEMV

## 技术路线

每个算子尽量包含以下实现与验证：

1. PyTorch Reference
2. CUDA Naive Kernel
3. CUDA Optimized Kernel
4. Correctness Test
5. Performance Benchmark
6. Optimization Report

## 当前进度

- [x] Jetson 基础环境检查
- [x] CUDA 编译与运行验证
- [x] PyTorch GPU 计算验证
- [x] PyTorch CUDA Extension 验证
- [x] 项目目录初始化
- [x] 项目基础代码框架
- [x] RMSNorm PyTorch Reference
- [x] RMSNorm CUDA Naive Kernel
- [x] RMSNorm CUDA Optimized Kernel
- [x] RMSNorm Optimization Report
- [x] RoPE PyTorch Reference
- [ ] RoPE CUDA Kernel
- [ ] INT8 Dequant-GEMV CUDA Kernel

## 项目目标

通过本项目系统学习和实践：

- CUDA 线程、线程块与网格划分
- Global Memory 与 Shared Memory
- Warp-Level Reduction
- 连续访存与向量化访问
- 算子融合
- CUDA Extension
- 正确性验证
- GPU 性能 Benchmark
- CUDA 到 TileLang / MXMACA 的迁移思路
