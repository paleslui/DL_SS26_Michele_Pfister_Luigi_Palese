"""Generic training loop for binary classification with class weighting and early stopping.

Used by every PyTorch model in the project (MLP, CNN, transformer). Each model
just provides its architecture; this module owns the optimization, validation,
device handling, and stopping logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -------- device autoselect --------------------------------------------------

def best_device() -> torch.device:
    """Return MPS (Apple Silicon GPU), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------- training config ----------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25            # early stopping patience on val AUC
    min_epochs: int = 20          # don't stop earlier than this
    seed: int = 42
    verbose: bool = False         # per-epoch printing


# -------- per-fold trainer ---------------------------------------------------

def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: torch.device
) -> DataLoader:
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _class_weight_pos(y_train: np.ndarray) -> float:
    """pos_weight for BCEWithLogitsLoss = (n_negative / n_positive)."""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig = TrainConfig(),
    device: torch.device | None = None,
) -> tuple[np.ndarray, dict]:
    """Train one model on one fold; return val probabilities and a history dict.

    Selects the model state with the best val AUC across all epochs (early stopping
    via patience, but always restores the best-AUC checkpoint at the end).
    """
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = device or best_device()
    model = model.to(device)

    train_loader = _make_loader(X_train, y_train, config.batch_size, True, device)
    val_loader = _make_loader(X_val, y_val, config.batch_size, False, device)

    pos_weight = torch.tensor(_class_weight_pos(y_train), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_auc = -1.0
    best_state: dict | None = None
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        # ---- train ----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        # ---- validate ----
        model.eval()
        val_losses, val_logits, val_targets = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                val_logits.append(logits.cpu().numpy())
                val_targets.append(yb.cpu().numpy())
        val_loss = float(np.mean(val_losses))
        val_logits_arr = np.concatenate(val_logits)
        val_targets_arr = np.concatenate(val_targets)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits_arr))
        try:
            val_auc = float(roc_auc_score(val_targets_arr, val_probs))
        except ValueError:
            val_auc = float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if config.verbose:
            print(f"    epoch {epoch:>3}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")

        # ---- early stopping & best-checkpoint tracking ----
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epoch >= config.min_epochs and epochs_without_improvement >= config.patience:
                break

    # Restore best weights and produce final val predictions
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_logits = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            all_logits.append(model(xb).cpu().numpy())
    final_logits = np.concatenate(all_logits)
    final_probs = 1.0 / (1.0 + np.exp(-final_logits))

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_auc
    return final_probs, history
