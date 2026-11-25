# CoLab 实验与笔记工作流模板

本项目演示 **本地编写代码/笔记 → GitHub 同步 → 浏览器 Colab 跑实验 → 结果自动回传本地** 的完整工作流，内置示例为《动手学深度学习》4.3 节 [多层感知机的简洁实现](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html)。

## 目录结构

```text
d:\CoLab\
├── configs/
│   └── mlp_concise.yaml        # 实验超参数配置
├── src/
│   └── experiments/
│       └── mlp_concise.py      # PyTorch 实验代码（本地/Colab 通用）
├── notebooks/
│   └── colab_runner.ipynb      # 浏览器 Colab 极简启动器
├── notes/
│   └── mlp_concise.md          # 实验笔记与结果记录
├── results/                    # 轻量实验产物（metrics.json / summary.md / 曲线图）
├── scripts/
│   └── sync.ps1                # Windows 一键拉取 & 推送 GitHub 脚本
├── requirements.txt
└── .gitignore                  # 自动忽略数据集与大权重文件 (*.pt/*.pth)
```

## 首次配置步骤（只需做一次）

1. **在 GitHub 创建一个新仓库**（例如 `CoLab-Experiments`，公开或私有均可）。
2. **在本地关联远程仓库并首次推送**：
   ```powershell
   git remote add origin https://github.com/<你的用户名>/<仓库名>.git
   .\scripts\sync.ps1 "init: setup colab workflow"
   ```
3. **在浏览器打开 Colab 启动器**：
   - 直接访问：`https://colab.research.google.com/github/<你的用户名>/<仓库名>/blob/main/notebooks/colab_runner.ipynb`
   - 或者在 Google Colab 里选择 **文件 → 打开笔记本 → GitHub**，输入你的仓库名打开 `notebooks/colab_runner.ipynb`。
   - 建议点击 **代码执行程序 → 更改运行时类型 → T4 GPU**。
4. **（可选，用于私有仓库或自动推回结果）配置 Colab Secrets**：
   - 在 GitHub 生成一个 Fine-grained Personal Access Token（给该仓库 `Contents: Read and write` 权限）。
   - 在 Colab 左侧边栏点 **钥匙图标（Secrets）**，新增 `GITHUB_TOKEN`，并打开“笔记本访问权限”。

## 日常使用流程

1. **本地开发与记录**：修改 `src/` 代码、`configs/` 超参数或 `notes/` 笔记。
2. **一键推送到 GitHub**：
   ```powershell
   .\scripts\sync.ps1 "exp: adjust hidden_dim to 512"
   ```
3. **浏览器 Colab 运行**：在 `notebooks/colab_runner.ipynb` 中修改 `GITHUB_REPO` 与 `CONFIG_PATH`，点击 **全部运行**。
   - 完整产物（含 `model.pt` 权重）会自动保存到你的 Google Drive（`/MyDrive/CoLab_Outputs/`），安装 Google Drive 桌面端即可自动同步回本地。
   - 轻量产物（`metrics.json`、`summary.md`、`learning_curve.png`）会自动复制进仓库 `results/`，如配置了 `GITHUB_TOKEN` 还会自动推送回 GitHub。
4. **本地拉取结果**：
   ```powershell
   .\scripts\sync.ps1
   ```
   （脚本会先自动 `git pull --rebase` 拉取 Colab 推回来的实验结果。）

## 本地直接运行测试

```powershell
python src/experiments/mlp_concise.py --config configs/mlp_concise.yaml --epochs 2
```
