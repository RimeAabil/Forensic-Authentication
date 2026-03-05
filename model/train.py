"""
train.py — Full training pipeline with early stopping, LR scheduling,
           TensorBoard logging, and best-model checkpointing.

Usage:
    python model/train.py
    python model/train.py --config path/to/config.yaml
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.dataset import AudioDataset
from model.cnn_lstm   import CNNLSTM

# ── Logging setup ─────────────────────────────────────────────────────────────
def setup_logging(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM forensic audio classifier")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


# ── Training / Validation loops ───────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    phase: str,
) -> tuple[float, float]:
    """
    Run one epoch of training or validation.

    Returns:
        (avg_loss, accuracy)
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for x, y in tqdm(loader, desc=f"  {phase.capitalize()}", leave=False, unit="batch"):
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(x)
            loss   = criterion(logits, y)

            if is_train:
                loss.backward()
                # Gradient clipping prevents exploding gradients in LSTM
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += y.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct   / max(total, 1)
    return avg_loss, accuracy


def main() -> None:
    args   = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_file = cfg["logging"]["log_file"]
    logger   = setup_logging(log_file)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Hyperparameters ───────────────────────────────────────────────────
    epochs        = cfg["training"]["epochs"]
    batch_size    = cfg["training"]["batch_size"]
    lr            = cfg["training"]["learning_rate"]
    wd            = cfg["training"]["weight_decay"]
    ckpt_dir      = cfg["training"]["checkpoint_dir"]
    patience      = cfg["training"]["early_stopping_patience"]
    splits_dir    = cfg["data"]["splits_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = AudioDataset(os.path.join(splits_dir, "train.csv"), augment=True)
    val_ds   = AudioDataset(os.path.join(splits_dir, "val.csv"),   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # ── Model, Loss, Optimiser ────────────────────────────────────────────
    model     = CNNLSTM(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6, verbose=True
    )

    # ── TensorBoard ───────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir  = os.path.join(cfg["logging"]["log_dir"], f"run_{run_id}")
    writer  = SummaryWriter(tb_dir)
    logger.info("TensorBoard logs: %s  →  tensorboard --logdir %s", tb_dir, cfg["logging"]["log_dir"])

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d / %d", epoch, epochs)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, "val")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # TensorBoard
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", current_lr, epoch)

        logger.info(
            "  Train  loss=%.4f  acc=%.4f | Val  loss=%.4f  acc=%.4f | LR=%.6f",
            train_loss, train_acc, val_loss, val_acc, current_lr,
        )

        # Best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            logger.info("  ✓ Best model saved → %s  (val_loss=%.4f)", best_path, val_loss)
        else:
            patience_count += 1
            logger.info("  No improvement (%d / %d)", patience_count, patience)

        # Periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  Checkpoint saved → %s", ckpt_path)

        # Early stopping
        if patience_count >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    writer.close()
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
