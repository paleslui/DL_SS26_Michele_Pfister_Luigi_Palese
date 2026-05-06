"""Generic training loop for binary classification.

Used by every PyTorch model in the project (MLP, CNN, transformer). Each model
just provides its architecture; this module owns the optimization, validation,
device handling, stopping logic, optional LR scheduling, and optional Optuna
pruning hooks for hyperparameter search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -------- device autoselect --------------------------------------------------

def best_device() -> torch.device:
    """Return MPS (Apple Silicon GPU), then CUDA, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------- training config ----------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25            # within-trial early stopping on val AUC plateau
    min_epochs: int = 20          # don't stop earlier than this

    # Optional LR scheduler — None / "cosine" / "plateau"
    scheduler_type: Optional[str] = None
    plateau_factor: float = 0.5
    plateau_patience: int = 10

    seed: int = 42
    verbose: bool = False         # per-epoch printing


# -------- helpers ------------------------------------------------------------

def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _class_weight_pos(y_train: np.ndarray) -> float:
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    return (n_neg / n_pos) if n_pos > 0 else 1.0


def _make_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if config.scheduler_type is None or config.scheduler_type == "none":
        return None
    if config.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.epochs)
        )
    if config.scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
        )
    raise ValueError(f"Unknown scheduler_type: {config.scheduler_type}")


# -------- per-fold trainer ---------------------------------------------------

def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig = TrainConfig(),
    device: torch.device | None = None,
) -> tuple[np.ndarray, dict]:
    """Train on one fold; return val probabilities and a history dict.

    Always restores the best-AUC checkpoint at the end (best practice).
    """
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = device or best_device()
    model = model.to(device)

    train_loader = _make_loader(X_train, y_train, config.batch_size, True)
    val_loader = _make_loader(X_val, y_val, config.batch_size, False)

    pos_weight = torch.tensor(_class_weight_pos(y_train), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, config)

    best_auc = -1.0
    best_state: dict | None = None
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
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

        # Step scheduler (plateau uses metric, cosine doesn't)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if config.verbose:
            print(f"    epoch {epoch:>3}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}",
                  flush=True)

        # Track best-checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epoch >= config.min_epochs and epochs_without_improvement >= config.patience:
                break

    # Restore best weights
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
