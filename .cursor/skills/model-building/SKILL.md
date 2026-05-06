---
name: model-building
description: Guides designing and implementing ML/DL model architectures, training loops, checkpoints, and experiment configs for research-style codebases. Use when the user mentions 模型搭建, 搭模型, model architecture, backbone, head, loss, optimizer, training loop, PyTorch nn.Module, or similar.
disable-model-invocation: true
---

# 模型搭建

## 何时启用

用户要**从零或从模板搭模型**、改结构、对齐张量形状、或整理训练/验证流程时，按本节顺序执行；不要跳过「规格对齐」。

## 工作流

### 1. 澄清任务与 I/O

- **任务**：分类 / 回归 / 序列预测 / 生成 / 表征学习 等。
- **输入**：形状（如 `[B, T, C]`）、模态、是否含缺失掩码。
- **输出**：标量、向量、序列、概率分布；是否需要不确定性或校准。
- **约束**：延迟、显存、是否必须可微、是否要导出 ONNX 等。

若用户未说明，先列出合理默认假设并标注为假设，避免静默猜错。

### 2. 基线与复杂度

- 先选**最简单可跑通**的基线（线性/小 MLP/轻量 CNN/小 Transformer 等），再按需加宽加深或加归纳偏置（卷积、RNN、注意力、图结构等）。
- 明确**为何**增加复杂度：指标瓶颈、归纳偏置匹配数据、或可解释性需求。

### 3. 模块与张量契约

- 为每个子模块写清 **in/out shape** 与 **dtype/device** 约定；跨模块边界处优先用显式 `view`/`permute`/`reshape` 并加注释说明物理含义（batch、时间、通道）。
- 损失与标签形状必须与模型输出一致；多任务时列出各 head 与对应 loss 的权重策略。

### 4. 训练骨架（最小可用）

建议默认包含：

- `model.train()` / `model.eval()` 切换
- 优化器与学习率策略（至少说明选用理由）
- 梯度裁剪（序列/大模型常见）是否在首轮启用
- **验证集**与早停/保存 best checkpoint 的判据
- 可复现：`seed`、确定性算子开关（若用户需要）

### 5. 调试顺序

1. 单 batch 前向 + 反传是否成功  
2. 损失是否下降、梯度是否非零  
3. 过拟合小数据集（sanity check）  
4. 再谈正则化、数据增强、结构搜索

## 代码风格

- 与仓库现有风格一致；无惯例时：**配置与常量**外置（yaml/dataclass），**计算图**放在 `nn.Module` 内，脚本只做组装与 CLI。
- 避免在 forward 里做重 I/O；日志用结构化标量（loss、lr、epoch、step）。

## 反模式

- 未对齐形状就堆模块导致运行时错误循环。
- 无验证集仅看训练 loss。
- 一上来大模型+复杂调度却无基线对照。

## 与本仓库主题（时间序列 / 缺失值）的提示

若涉及**时间序列补全或缺失掩码**：

- 明确缺失机制（MCAR/MAR/MNAR）是否建模；掩码如何进入网络（concat 通道、单独分支、attention bias 等）。
- 评估指标需与「仅对缺失位置」或「全序列」一致，并在代码与文档中统一。
