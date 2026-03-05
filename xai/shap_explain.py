"""
shap_explain.py — SHAP-based feature importance for court reports.

Uses GradientExplainer (faster than KernelExplainer for PyTorch models)
to compute the contribution of each LFCC coefficient to the tampering
prediction.  The output is a bar chart ranking coefficients by importance.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.dataset      import AudioDataset
from features.extract_lfcc import extract_lfcc, N_LFCC
from model.cnn_lstm        import load_model

logger = logging.getLogger(__name__)

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)


def build_background(
    csv_path: str,
    n_samples: int = 50,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a background dataset for SHAP from training samples.
    Background represents the "expected" distribution the explainer
    compares against.

    Args:
        csv_path  : Path to train.csv.
        n_samples : Number of background samples (50 is sufficient).
        device    : Compute device.

    Returns:
        Tensor of shape (n_samples, 1, N_LFCC, T).
    """
    device  = device or torch.device("cpu")
    dataset = AudioDataset(csv_path, augment=False)
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    samples = [dataset[i][0] for i in indices]    # list of (1, N_LFCC, T) tensors
    return torch.stack(samples).to(device)


def explain_prediction(
    audio_path: str,
    output_path: str = "xai/shap_output.png",
    model_path: str | None = None,
    device: torch.device | None = None,
    n_background: int = 50,
) -> tuple[np.ndarray, str]:
    """
    Compute per-LFCC-coefficient SHAP importances and save a bar chart.

    Args:
        audio_path   : Audio file to explain.
        output_path  : Destination PNG path.
        model_path   : Override config model path.
        device       : Override compute device.
        n_background : Number of background samples for SHAP explainer.

    Returns:
        (importance_array of shape (N_LFCC,), output_path)
    """
    device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path or _cfg["app"]["model_path"]
    splits_dir = _cfg["data"]["splits_dir"]

    model = load_model(_cfg, model_path, device)

    # Build background
    train_csv  = os.path.join(splits_dir, "train.csv")
    background = build_background(train_csv, n_samples=n_background, device=device)

    # Prepare input
    lfcc = extract_lfcc(audio_path)
    x    = torch.tensor(lfcc).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,N,T)

    # GradientExplainer — uses expected gradients
    explainer = shap.GradientExplainer(model, background)
    shap_vals = explainer.shap_values(x)   # list of arrays, one per class

    # shap_vals[1] → tampered class; shape: (1, 1, N_LFCC, T)
    tampered_shap = np.abs(shap_vals[1][0, 0])   # (N_LFCC, T)
    importance    = tampered_shap.mean(axis=1)     # (N_LFCC,) — average over time

    # ── Plot ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    sorted_idx = np.argsort(importance)[::-1]
    labels     = [f"LFCC-{i:02d}" for i in sorted_idx]
    values     = importance[sorted_idx]

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP value|  (contribution to 'Tampered' prediction)", fontsize=11)
    ax.set_title("SHAP Feature Importance — Which LFCC Coefficients Drove the Decision",
                 fontsize=12, fontweight="bold")
    ax.axvline(x=importance.mean(), color="steelblue", linestyle="--",
               linewidth=1.5, label=f"Mean = {importance.mean():.4f}")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("SHAP plot saved → %s", output_path)
    return importance, output_path
