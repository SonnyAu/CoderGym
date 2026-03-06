"""
MLP with Gradient Clipping and Early Stopping (Training & Optimization L3)

Train an MLP with torch.nn.utils.clip_grad_norm_ and early stopping on validation loss.
Saves best model state in save_artifacts.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "train_opt_lvl3_gradient_clip_early_stop",
        "task_type": "classification",
        "description": "MLP with gradient clipping and early stopping; save best model.",
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the device for computation."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    n_samples: int = 800,
    n_features: int = 10,
    n_classes: int = 3,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    device: torch.device = None,
) -> Tuple[DataLoader, DataLoader]:
    """Synthetic blobs; train/val split."""
    if device is None:
        device = get_device()
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=1.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - train_ratio, random_state=42, stratify=y)
    # Normalize
    mu, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_loader, val_loader


class MLP(nn.Module):
    """Small MLP for classification."""

    def __init__(self, input_dim: int, hidden: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


def build_model(input_dim: int, num_classes: int = 3, lr: float = 0.01, device: torch.device = None) -> Tuple[nn.Module, optim.Optimizer]:
    """Build MLP and optimizer."""
    if device is None:
        device = get_device()
    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    max_epochs: int,
    patience: int = 10,
    max_norm: float = 1.0,
    print_every: int = 5,
) -> Tuple[list, int, float, Dict[str, Any]]:
    """
    Train with gradient clipping (clip_grad_norm_) and early stopping on validation loss.
    Returns (loss_history, best_epoch, best_val_metric, best_state_dict).
    """
    device = model.get_device()
    loss_history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_val_metric = 0.0
    epochs_without_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        # Validation loss for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item()
        val_loss /= len(val_loader)
        # Val accuracy as best_val_metric
        val_metrics = evaluate(model, val_loader, criterion, task_type="classification")
        acc = val_metrics["accuracy"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_val_metric = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}")
        if epochs_without_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break
    return loss_history, best_epoch, best_val_metric, best_state


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    task_type: str = "classification",
) -> Dict[str, float]:
    """Returns MSE/R2 for regression or accuracy/loss for classification."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(model.get_device())
            y = y.to(model.get_device())
            out = model(X)
            if task_type == "classification":
                loss = criterion(out, y)
                total_loss += loss.item()
                _, pred = out.max(1)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
            else:
                total_loss += criterion(out, y).item()
                all_preds.append(out.cpu().numpy())
                all_targets.append(y.cpu().numpy())
            n += 1
    total_loss /= n
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    if task_type == "classification":
        acc = accuracy_score(targets, preds)
        return {"loss": total_loss, "accuracy": acc}
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    return {"loss": total_loss, "mse": mse, "r2": r2}


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Predict class labels."""
    model.eval()
    t = torch.FloatTensor(X).to(model.get_device())
    with torch.no_grad():
        out = model(t)
        pred = out.argmax(1)
    return pred.cpu().numpy()


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    loss_history: list,
    best_epoch: int,
    best_val_metric: float,
    best_state_dict: Optional[Dict],
    output_dir: str = "output",
) -> None:
    """Save best model state, metrics, and history."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")
    out = {
        "metrics": metrics,
        "loss_history": loss_history,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
    }
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Artifacts saved to {output_dir} (best model from epoch {best_epoch})")


def main() -> int:
    """Train with clipping and early stopping; assert training terminates and save best model."""
    print("=" * 60)
    print("Training & Optimization L3: Gradient Clipping + Early Stopping")
    print("=" * 60)
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders(n_samples=800, train_ratio=0.8, batch_size=32, device=device)
    input_dim = 10
    num_classes = 3
    model, optimizer = build_model(input_dim=input_dim, num_classes=num_classes, lr=0.01, device=device)
    criterion = nn.CrossEntropyLoss()

    loss_history, best_epoch, best_val_metric, best_state = train(
        model, train_loader, val_loader, optimizer, criterion,
        max_epochs=200, patience=10, max_norm=1.0, print_every=10,
    )

    # Load best for final eval
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_loader, criterion, task_type="classification")
    val_metrics = evaluate(model, val_loader, criterion, task_type="classification")

    print("\nEpochs run:", len(loss_history))
    print("Best epoch:", best_epoch, "Best val accuracy:", best_val_metric)
    print("Train — Loss: {:.4f}, Accuracy: {:.4f}".format(train_metrics["loss"], train_metrics["accuracy"]))
    print("Val   — Loss: {:.4f}, Accuracy: {:.4f}".format(val_metrics["loss"], val_metrics["accuracy"]))

    save_artifacts(
        model,
        {"train": train_metrics, "validation": val_metrics},
        loss_history,
        best_epoch,
        best_val_metric,
        best_state,
        output_dir="output",
    )

    # Assert training terminated (early stop or ran full epochs)
    if len(loss_history) == 0:
        print("\nFAIL: No training steps run")
        return 1
    print("\nPASS: Training terminated; best model saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
