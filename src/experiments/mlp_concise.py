"""D2L 4.3 多层感知机的简洁实现 (PyTorch / Fashion-MNIST).

参考: https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(input_dim: int, hidden_dim: int, output_dim: int, init_std: float) -> nn.Module:
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )

    def init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=init_std)

    net.apply(init_weights)
    return net


def get_dataloaders(
    data_dir: Path, batch_size: int, num_workers: int, pin_memory: bool
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(
        root=str(data_dir), train=True, transform=transform, download=True
    )
    test_dataset = datasets.FashionMNIST(
        root=str(data_dir), train=False, transform=transform, download=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate_accuracy(net: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    net.eval()
    correct = 0
    total = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        preds = net(X).argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


def train_one_epoch(
    net: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    net.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        batch_n = y.numel()
        loss_sum += float(loss.item()) * batch_n
        correct += int((y_hat.argmax(dim=1) == y).sum().item())
        total += batch_n

    return loss_sum / max(total, 1), correct / max(total, 1)


def plot_learning_curve(history: list[dict[str, Any]], save_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    test_acc = [row["test_acc"] for row in history]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    ax.plot(epochs, train_loss, label="train loss", linestyle="-", marker="o", markersize=4)
    ax.plot(epochs, train_acc, label="train acc", linestyle="--", marker="s", markersize=4)
    ax.plot(epochs, test_acc, label="test acc", linestyle="-.", marker="^", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("MLP Concise (Fashion-MNIST)")
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0.0, max(1.05, max(train_loss) * 1.1))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def write_summary_md(report: dict[str, Any], save_path: Path) -> None:
    cfg = report["config"]
    final = report["final_metrics"]
    lines = [
        f"# 实验报告：{cfg['experiment_name']}",
        "",
        f"- **运行时间**：`{report['timestamp']}`",
        f"- **运行设备**：`{report['device']}`",
        f"- **总耗时**：`{report['elapsed_seconds']:.2f}s`",
        f"- **最终指标**：`train_loss={final['train_loss']:.4f}` | "
        f"`train_acc={final['train_acc']:.4f}` | `test_acc={final['test_acc']:.4f}`",
        "",
        "## 超参数配置",
        "",
        f"- `hidden_dim`: `{cfg['model']['hidden_dim']}`",
        f"- `init_std`: `{cfg['model']['init_std']}`",
        f"- `batch_size`: `{cfg['training']['batch_size']}`",
        f"- `lr`: `{cfg['training']['lr']}`",
        f"- `num_epochs`: `{cfg['training']['num_epochs']}`",
        f"- `seed`: `{cfg['training']['seed']}`",
        "",
        "## 训练过程",
        "",
        "| Epoch | Train Loss | Train Acc | Test Acc | Time (s) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in report["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.4f} | "
            f"{row['train_acc']:.4f} | {row['test_acc']:.4f} | {row['epoch_seconds']:.2f} |"
        )
    lines.extend(["", "![Learning Curve](./learning_curve.png)", ""])
    save_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D2L MLP Concise experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mlp_concise.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store/load Fashion-MNIST dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save outputs. Defaults to results/<exp_name>/<timestamp>.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training.num_epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip saving model.pt checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path("results") / cfg["experiment_name"] / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    device_label = (
        f"cuda ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"
    )
    print(f"[INFO] Experiment: {cfg['experiment_name']}")
    print(f"[INFO] Device: {device_label}")
    print(f"[INFO] Output dir: {output_dir.resolve()}")

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
    )

    net = build_model(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        output_dim=int(cfg["model"]["output_dim"]),
        init_std=float(cfg["model"]["init_std"]),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=float(cfg["training"]["lr"]))

    history: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    num_epochs = int(cfg["training"]["num_epochs"])

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(net, train_loader, loss_fn, optimizer, device)
        test_acc = evaluate_accuracy(net, test_loader, device)
        epoch_sec = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "epoch_seconds": epoch_sec,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | {epoch_sec:.2f}s"
        )

    elapsed_all = time.perf_counter() - start_all
    report = {
        "timestamp": timestamp,
        "device": device_label,
        "elapsed_seconds": elapsed_all,
        "config": cfg,
        "final_metrics": history[-1],
        "history": history,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_learning_curve(history, output_dir / "learning_curve.png")
    write_summary_md(report, output_dir / "summary.md")

    if not args.no_save_model:
        torch.save(net.state_dict(), output_dir / "model.pt")

    print(f"[DONE] Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
