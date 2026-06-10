# FRINET — 过参数化连续原子稀疏恢复

基于深度学习的 FRI（Finite Rate of Innovation）信号脉冲参数估计。

## 文件结构

```
NN_FRI/
├── fri_net.py                       ContinuousAtomNet 模型定义（过参数化稀疏恢复）
├── fri_generator.py                 FRI 信号生成器（高斯/指数脉冲，可控 SNR）
├── batch_runner.py                  核心训练引擎：逐信号梯度下降优化（被各实验调用）
├── traditional_methods.py           传统方法（矩阵束法 Matrix Pencil）
├── train.py                         监督学习训练脚本（使用 PythonProject1 的数据和模型）
├── eval.py                          评估脚本（待完善）
│
├── experiment_1_baseline.py         实验 1：基线测试（50 信号快速验证框架）
├── hyperparam_search.py             实验 2：超参数搜索（M, λ, lr 最优值）
├── experiment_overparam.py          实验 3：过参数化效果研究（M 的影响）
├── 11experiment_snr_comparison.py   实验 4：SNR 对比扫 -5~30dB（FRINET vs 矩阵束法）
├── 11failure_analysis.py            实验 5：失败案例分析（0dB / 10dB 失效模式）
├── 12sensitivity_analysis.py        实验 6：初始化 & 正则化灵敏度分析
├── 12robustness_tests.py            实验 7：泛化鲁棒性测试（不同脉冲类型/宽度/K/SNR）
│
├── PythonProject1/                  旧版：监督学习方案
│   ├── data_generation.py           数据生成（K=3, N=100, 10000 样本）
│   ├── model_arch.py                FRINet 模型（编码器 + 残差块 + 双头输出）
│   ├── train.py                     训练脚本
│   ├── benchmark_classic.py         经典方法基准
│   └── eval_experiments.py          实验评估
│
├── datasets/                        数据集目录
├── models/                          模型保存目录
├── results/                         结果图输出目录
└── picture/                         图片目录
```

## 正常实验顺序

| 步骤 | 运行文件 | 说明 |
|------|----------|------|
| ① 快速验证 | `python experiment_1_baseline.py` | 50 信号，确认框架能跑通 |
| ② 超参数搜索 | `python hyperparam_search.py` | 搜索最优 M、λ、lr |
| ③ 过参数化 | `python experiment_overparam.py` | 验证 M 增大对性能的影响 |
| ④ SNR 对比 | `python 11experiment_snr_comparison.py` | -5~30dB，对比矩阵束法 |
| ⑤ 失败分析 | `python 11failure_analysis.py` | 0dB/10dB 失效模式诊断 |
| ⑥ 灵敏度 | `python 12sensitivity_analysis.py` | 初始化灵敏度、λ 敏感性 |
| ⑦ 鲁棒性 | `python 12robustness_tests.py` | 脉冲类型/宽度/K/SNR 泛化测试 |

## 运行结果

### ① 基线测试 (K=3, SNR=25dB, 50 信号)

| 指标 | FRINET | 矩阵束法 |
|------|--------|----------|
| 时延 RMSE | **0.01790** | 0.48539 |
| 成功率 | **34.0%** | — |

> FRINET 时延估计精度远超矩阵束法，但成功率有提升空间。
50信号基线测试.png

### ② 超参数搜索

网格搜索 M×λ 组合
网格搜索找到最优参数组合.png

### ③ 过参数化效果

不同 SNR 下，观察成功率/RMSE 随 M 变化
固定 SNR，观察成功率如何随 M 变化.png
### ④ SNR 对比 (-5 ~ 30dB)

FRINET 与矩阵束法在不同信噪比下的精度和成功率对比。
全 SNR 范围对比实验.png

### ⑤ 失败案例分析 (0dB / 10dB)

分析低 SNR 下的主要失效模式（脉冲数错误、时延偏差大）。
失败分析 SNR=0dB.png/失败分析SNR10dB.png

### ⑥ 灵敏度分析

| 指标 | 结果 |
|------|------|
| 20 次不同初始化，时延标准差 | **0.02067** |

> 初始化影响很小，模型收敛稳定。
灵敏度分析_对初始化对 λ 的敏感性.png

### ⑦ 鲁棒性测试 (9 种场景，每种 30 信号)

| 测试场景 | RMSE | 成功率 |
|----------|------|--------|
| 基准高斯 (σ=0.08, K=3, SNR=20) | 0.02136 | 30.0% |
| 宽高斯 (σ=0.15) | 0.07809 | 6.7% |
| 窄高斯 (σ=0.04) | **0.00543** | **50.0%** |
| 指数衰减 | 0.12583 | 0.0% |
| K=2 | 0.00523 | 53.3% |
| K=5 | 0.14626 | 3.3% |
| K=2 低 SNR (5dB) | 0.13239 | 10.0% |
| K=5 低 SNR (5dB) | 0.19715 | 0.0% |
| 密集脉冲 | 0.05754 | 20.0% |
泛化测试矩阵.png

**关键发现：**
- 窄高斯 + 基准 K=3 表现最好（RMSE 0.005, 成功率 50%）
- K 增大显著降低成功率（K=5 仅 3.3%）
- 指数衰减脉冲完全不适用当前高斯模型
- 低 SNR 是主要挑战

## 核心概念

- **逐信号优化**：对每个含噪信号独立运行梯度下降，找到最优的 τ 和 a
- **过参数化**：M >> K（如 M=100, K=3），用 ℓ₁ 正则化推动稀疏
- **聚类后处理**：合并邻近原子，避免同一脉冲被多个原子表示
