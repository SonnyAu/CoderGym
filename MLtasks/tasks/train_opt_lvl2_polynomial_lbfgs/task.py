"""
Polynomial Regression with L-BFGS (Training & Optimization L2)

L-BFGS is a quasi-Newton method that uses a closure-based interface in PyTorch:
the optimizer repeatedly calls the closure to recompute the loss and gradients
for the current parameters, enabling line search. This often converges in fewer
epochs than SGD for small, smooth objectives.

Fit: y = 1 + 2*x + 0.5*x^2 + noise with manual polynomial features.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Any, Tuple

# sklearn for metrics only
from sklearn.metrics import mean_squared_error, r2_score


def get_task_metadata() -> Dict[str, Any]:
    """Return task metadata."""
    return {
        "task_name": "train_opt_lvl2_polynomial_lbfgs",
        "task_type": "regression",
        "description": "Polynomial regression with L-BFGS (quasi-Newton, closure-based).",
        "true_coeffs": [1.0, 2.0, 0.5],
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
    n_train: int = 400,
    n_val: int = 100,
    noise_std: float = 0.3,
    degree: int = 2,
    batch_size: int = 400,
    device: torch.device = None,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic: y = 1 + 2*x + 0.5*x^2 + noise.
    Polynomial features: [1, x, x^2]. Train/val split.
    """
    if device is None:
        device = get_device()
    # True coefficients [1, 2, 0.5]
    true_coeffs = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    x_train = np.random.uniform(-3, 3, size=n_train)
    x_val = np.random.uniform(-3, 3, size=n_val)
    X_train = np.column_stack([x_train**i for i in range(degree + 1)])
    X_val = np.column_stack([x_val**i for i in range(degree + 1)])
    y_train = X_train @ true_coeffs + np.random.randn(n_train).astype(np.float32) * noise_std
    y_val = X_val @ true_coeffs + np.random.randn(n_val).astype(np.float32) * noise_std

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).unsqueeze(1).to(device),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val).to(device),
        torch.FloatTensor(y_val).unsqueeze(1).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, n_train), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=n_val, shuffle=False)
    return train_loader, val_loader, X_train, X_val, y_train, y_val


class PolynomialModel(nn.Module):
    """Linear layer: input_dim (degree+1) -> 1."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


def build_model(input_dim: int, device: torch.device = None) -> nn.Module:
    """Build polynomial regression model."""
    if device is None:
        device = get_device()
    return PolynomialModel(input_dim).to(device)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    epochs: int = 50,
    lr: float = 1.0,
) -> Tuple[list, int]:
    """
    Train using L-BFGS (closure-based). No SGD.
    Returns (loss_history, epochs_run).
    """
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    loss_history = []
    epochs_run = 0
    device = model.get_device()
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            # Full batch for L-BFGS is typical
            X, y = next(iter(train_loader))
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if loss is not None:
            loss_val = loss.item()
        else:
            loss_val = float("nan")
        loss_history.append(loss_val)
        epochs_run += 1
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_val:.6f}")
        # Early stop if converged
        if len(loss_history) >= 2 and abs(loss_history[-1] - loss_history[-2]) < 1e-8:
            break
    return loss_history, epochs_run


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Compute MSE and R2 on the given loader."""
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(model.get_device())
            y = y.to(model.get_device())
            out = model(X)
            total_loss += criterion(out, y).item()
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
            n += 1
    preds = np.vstack(preds).ravel()
    targets = np.vstack(targets).ravel()
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    return {"loss": total_loss / n, "mse": mse, "r2": r2}


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Predict for numpy X (n, input_dim)."""
    model.eval()
    t = torch.FloatTensor(X).to(model.get_device())
    with torch.no_grad():
        out = model(t)
    return out.cpu().numpy().ravel()


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    loss_history: list,
    output_dir: str = "output",
) -> None:
    """Save model state and metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(output_dir) / "model.pt")
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "loss_history": loss_history}, f, indent=2)
    print(f"Artifacts saved to {output_dir}")


def main() -> int:
    """Train with L-BFGS, assert R2 > 0.9, print learned coefficients."""
    print("=" * 60)
    print("Training & Optimization L2: Polynomial Regression with L-BFGS")
    print("=" * 60)
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    n_train, n_val = 400, 100
    degree = 2
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_train=n_train, n_val=n_val, degree=degree, batch_size=n_train, device=device
    )
    input_dim = X_train.shape[1]
    model = build_model(input_dim, device=device)
    criterion = nn.MSELoss()

    loss_history, epochs_run = train(model, train_loader, criterion, epochs=50, lr=1.0)
    train_metrics = evaluate(model, train_loader, criterion)
    val_metrics = evaluate(model, val_loader, criterion)

    # Learned coefficients (model.linear.weight, model.linear.bias)
    with torch.no_grad():
        w = model.linear.weight.cpu().numpy().ravel()
        b = model.linear.bias.cpu().item()
    learned = np.concatenate([[b], w])
    true_coeffs = np.array([1.0, 2.0, 0.5])
    print("\nLearned coefficients:", learned)
    print("True coefficients:  ", true_coeffs)
    print("\nTrain — Loss: {:.4f}, MSE: {:.4f}, R2: {:.4f}".format(
        train_metrics["loss"], train_metrics["mse"], train_metrics["r2"]))
    print("Val   — Loss: {:.4f}, MSE: {:.4f}, R2: {:.4f}".format(
        val_metrics["loss"], val_metrics["mse"], val_metrics["r2"]))
    # Convergence: L-BFGS typically needs fewer epochs than SGD for this problem (comment in code)
    # E.g. SGD might need 100+ epochs; L-BFGS often converges in tens of steps.

    save_artifacts(model, {"train": train_metrics, "validation": val_metrics, "learned_coeffs": learned.tolist()}, loss_history)

    if val_metrics["r2"] <= 0.9:
        print(f"\nFAIL: Validation R2 {val_metrics['r2']:.4f} is not > 0.9")
        return 1
    print("\nPASS: Validation R2 > 0.9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
