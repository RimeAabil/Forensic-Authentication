"""
evaluate.py — Full evaluation pipeline.

Computes:
    - Classification report (precision, recall, F1)
    - Confusion matrix
    - EER (Equal Error Rate)   — primary forensic metric
    - Precision @ 1% FAR       — judicial admissibility metric
    - ROC curve plot

Usage:
    python model/evaluate.py
    python model/evaluate.py --config config.yaml --split test
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data  import DataLoader
from sklearn.metrics   import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.dataset import AudioDataset
from model.cnn_lstm   import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forensic audio model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split",  default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="evaluation")
    return parser.parse_args()


def compute_eer(fpr: np.ndarray, fnr: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate: the point where FPR ≈ FNR.

    Returns:
        (eer_value, eer_threshold_index)
    """
    diff = np.abs(fpr - fnr)
    idx  = int(np.argmin(diff))
    eer  = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, idx


def precision_at_far(tpr: np.ndarray, fpr: np.ndarray, target_far: float = 0.01) -> float:
    """
    Return the True Positive Rate (sensitivity) at a given False Alarm Rate.
    In the judicial context, FAR = 1% means we allow only 1 false accusation
    per 100 authentic samples.
    """
    diffs = np.abs(fpr - target_far)
    idx   = int(np.argmin(diffs))
    return float(tpr[idx])


def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Authentic", "Tampered"],
        yticklabels=["Authentic", "Tampered"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved → %s", output_path)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
             eer: float, eer_idx: int, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.scatter(fpr[eer_idx], tpr[eer_idx], color="red", zorder=5,
               label=f"EER = {eer*100:.2f}%")
    ax.axvline(x=0.01, color="orange", linestyle="--", lw=1, label="1% FAR threshold")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — Forensic Audio Authentication")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("ROC curve saved → %s", output_path)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────
    model_path = cfg["app"]["model_path"]
    if not os.path.exists(model_path):
        logger.error("Model not found at %s — run train.py first", model_path)
        sys.exit(1)

    model = load_model(cfg, model_path, device)

    # ── Load data ─────────────────────────────────────────────────────────
    csv_path = os.path.join(cfg["data"]["splits_dir"], f"{args.split}.csv")
    dataset  = AudioDataset(csv_path, augment=False)
    loader   = DataLoader(dataset, batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)
    logger.info("Evaluating on %d samples from %s", len(dataset), csv_path)

    # ── Inference ─────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for x, y in loader:
            x      = x.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ── Metrics ───────────────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=["Authentic", "Tampered"],
        digits=4,
    )
    logger.info("\n%s", report)

    cm = confusion_matrix(all_labels, all_preds)
    logger.info("Confusion Matrix:\n%s", cm)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fnr     = 1.0 - tpr
    roc_auc = auc(fpr, tpr)

    eer, eer_idx = compute_eer(fpr, fnr)
    p_at_1far    = precision_at_far(tpr, fpr, target_far=0.01)

    logger.info("─" * 50)
    logger.info("AUC            : %.4f", roc_auc)
    logger.info("EER            : %.2f%%  (target < 5%%)", eer * 100)
    logger.info("Precision@1%%FAR: %.2f%%", p_at_1far * 100)
    logger.info("─" * 50)

    # ── Save plots ────────────────────────────────────────────────────────
    plot_confusion_matrix(
        cm, os.path.join(args.output_dir, "confusion_matrix.png")
    )
    plot_roc(
        fpr, tpr, roc_auc, eer, eer_idx,
        os.path.join(args.output_dir, "roc_curve.png"),
    )

    # ── Save summary text ─────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(report)
        f.write(f"\nAUC             : {roc_auc:.4f}\n")
        f.write(f"EER             : {eer*100:.2f}%\n")
        f.write(f"Precision@1%FAR : {p_at_1far*100:.2f}%\n")

    logger.info("Evaluation summary saved → %s", summary_path)


if __name__ == "__main__":
    main()
