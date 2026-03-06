"""
MLP on Fashion-MNIST (Training & Optimization L1)
Train a feedforward MLP on Fashion-MNIST with train/val split; report accuracy and loss.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from torchvision import datasets, transforms  # type: ignore
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "train_opt_lvl1_fashion_mnist_mlp",
        "task_type": "classification",
        "num_classes": 10,
        "input_dim": 784,
        "description": "MLP on Fashion-MNIST with Adam; device-agnostic.",
        "metrics": ["accuracy", "loss", "per_class_accuracy"],
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the device for computation (cuda/cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    batch_size: int = 64,
    val_ratio: float = 0.2,
    data_root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders for Fashion-MNIST.
    80/20 train/val split from the official training set.
    Falls back to synthetic data if torchvision is not installed.
    """
    if HAS_TORCHVISION:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        full_train = datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=transform
        )
        val_size = int(len(full_train) * val_ratio)
        train_size = len(full_train) - val_size
        train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        return train_loader, val_loader
    # Fallback: learnable synthetic data (blobs in 784-d) when torchvision not installed
    from sklearn.datasets import make_blobs
    set_seed(42)
    n_total = 5000
    n_train = int(n_total * (1 - val_ratio))
    n_val = n_total - n_train
    X, y = make_blobs(n_samples=n_total, n_features=784, centers=10, cluster_std=2.0, random_state=42)
    X = X.astype(np.float32).reshape(n_total, 1, 28, 28)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Synthetic data (no torchvision). Training samples: {n_train}, Validation samples: {n_val}")
    return train_loader, val_loader


class MLP(nn.Module):
    """Feedforward MLP for Fashion-MNIST (28x28 -> 784 -> hidden -> 10)."""

    def __init__(self, input_dim: int = 784, hidden_dims: Tuple[int, ...] = (256, 128), num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


def build_model(
    input_dim: int = 784,
    hidden_dims: Tuple[int, ...] = (256, 128),
    num_classes: int = 10,
    lr: float = 0.001,
    device: torch.device = None,
) -> Tuple[nn.Module, optim.Optimizer]:
    """Build MLP and Adam optimizer."""
    if device is None:
        device = get_device()
    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    print_every: int = 5,
) -> list:
    """Train the model; return loss_history (per epoch)."""
    model.train()
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(model.get_device()), y.to(model.get_device())
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        avg_loss = running_loss / n_batches
        loss_history.append(avg_loss)
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")
    return loss_history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate on a dataloader. Returns accuracy, mean loss, and per-class accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(model.get_device()), y.to(model.get_device())
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()
            _, pred = logits.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            for c in range(num_classes):
                idx = (y == c)
                class_total[c] += idx.sum().item()
                class_correct[c] += (pred[idx] == y[idx]).sum().item()
    n_batches = len(data_loader)
    mean_loss = total_loss / n_batches
    accuracy = correct / total if total else 0.0
    per_class_acc = [class_correct[c] / class_total[c] if class_total[c] else 0.0 for c in range(num_classes)]
    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
    }


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Predict class labels for input X (n, 1, 28, 28 or n, 784)."""
    model.eval()
    dev = model.get_device()
    if X.ndim == 2 and X.shape[1] == 784:
        t = torch.FloatTensor(X).to(dev)
    else:
        t = torch.FloatTensor(X).to(dev)
    with torch.no_grad():
        logits = model(t)
        pred = logits.argmax(1)
    return pred.cpu().numpy()


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    loss_history: list,
    val_accuracy_history: list,
    output_dir: str = "output",
) -> None:
    """Save model state, metrics, and optional history."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(output_dir) / "model.pt")
    out = {
        "metrics": metrics,
        "loss_history": loss_history,
        "val_accuracy_history": val_accuracy_history,
    }
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Artifacts saved to {output_dir}")


def main() -> int:
    """Train, evaluate, assert validation accuracy > 0.82; exit 0 on success."""
    print("=" * 60)
    print("Training & Optimization L1: MLP on Fashion-MNIST")
    print("=" * 60)
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    batch_size = 64
    val_ratio = 0.2
    epochs = 15
    lr = 0.001

    train_loader, val_loader = make_dataloaders(batch_size=batch_size, val_ratio=val_ratio)
    model, optimizer = build_model(lr=lr, device=device)
    criterion = nn.CrossEntropyLoss()

    loss_history = train(model, train_loader, optimizer, criterion, epochs=epochs)
    val_accuracy_history = []
    model.eval()
    for _ in range(1):
        m = evaluate(model, val_loader, criterion)
        val_accuracy_history.append(m["accuracy"])

    train_metrics = evaluate(model, train_loader, criterion)
    val_metrics = evaluate(model, val_loader, criterion)

    print("\n--- Train metrics ---")
    print(f"  Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
    print("--- Validation metrics ---")
    print(f"  Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
    print("  Per-class accuracy:", [f"{a:.3f}" for a in val_metrics["per_class_accuracy"]])

    save_artifacts(model, {"train": train_metrics, "validation": val_metrics}, loss_history, val_accuracy_history)

    # Assert validation accuracy > 0.82
    if val_metrics["accuracy"] <= 0.82:
        print(f"\nFAIL: Validation accuracy {val_metrics['accuracy']:.4f} is not > 0.82")
        return 1
    print("\nPASS: Validation accuracy > 0.82")
    return 0


if __name__ == "__main__":
    sys.exit(main())
