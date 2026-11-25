# 4.3 多层感知机的简洁实现（笔记）

- 原文链接：https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html
- 对应代码：`src/experiments/mlp_concise.py`
- 对应配置：`configs/mlp_concise.yaml`

## 核心结构

使用 PyTorch 高级 API `nn.Sequential` 构建含单个隐藏层的多层感知机（MLP），用于 Fashion-MNIST 10 分类任务：

1. `nn.Flatten()`：将 `28×28` 图像展平为 `784` 维向量。
2. `nn.Linear(784, 256)` + `nn.ReLU()`：隐藏层与非线性激活函数。
3. `nn.Linear(256, 10)`：输出层（直接输出 logits，配合 `nn.CrossEntropyLoss()` 内部做 Softmax）。
4. 权重初始化：对所有 `nn.Linear` 层使用均值 `0`、标准差 `0.01` 的正态分布初始化。

## 关键要点

- **与 softmax 回归的唯一区别**：多了一个全连接隐藏层和 ReLU 激活函数，模型表达能力显著提升。
- **工程实践**：
  - 将超参数抽离到 `configs/mlp_concise.yaml`，便于后续做消融实验（例如对比 `hidden_dim=128/256/512`、`lr=0.05/0.1` 或更换激活函数）。
  - 训练脚本同时输出 `metrics.json`、`summary.md` 和 `learning_curve.png`，Colab 跑完即可自动同步回本地。

## 实验记录

| 日期 | 配置变更 | 设备 | Train Loss | Train Acc | Test Acc | 结果目录 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| _待补充_ | 基线 (`hidden_dim=256, lr=0.1, epochs=10`) | Colab T4 | - | - | - | `results/mlp_concise/...` |
