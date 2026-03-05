"""
gradcam.py — Grad-CAM visualization for the CNN-LSTM forensic model.

Highlights which regions of the LFCC spectrogram the model considers
most suspicious — essential for generating court-admissible explanations.

Grad-CAM works by computing the gradient of the target class score
with respect to the final convolutional feature maps.  High-gradient
regions correspond to frequency-time areas that strongly influenced
the tampering prediction.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless backend — safe for servers
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.extract_lfcc import extract_lfcc
from model.cnn_lstm        import load_model

logger = logging.getLogger(__name__)

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)


class GradCAM:
    """
    Grad-CAM implementation targeting the last Conv2d layer of the CNN backbone.

    Args:
        model        : CNNLSTM instance in eval mode.
        target_layer : The nn.Module (Conv2d) to hook.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            x            : Input tensor (1, 1, N_LFCC, T).
            target_class : Class index to explain (1 = tampered).

        Returns:
            2-D numpy array — heatmap of shape matching the feature map spatial dims.
        """
        self.model.zero_grad()
        x = x.requires_grad_(True)
        logits = self.model(x)

        # Backprop only for the target class score
        score = logits[0, target_class]
        score.backward()

        # Pool gradients across channels
        weights     = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam         = (weights * self.activations).sum(dim=1)         # (1, H, W)
        cam         = torch.relu(cam).squeeze().cpu().numpy()         # ReLU → keep positive influence

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam


def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Retrieve the last Conv2d inside the CNN backbone."""
    last_conv = None
    for module in model.cnn.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d found in model.cnn")
    return last_conv


def generate_gradcam_plot(
    audio_path: str,
    output_path: str = "xai/gradcam_output.png",
    model_path: str | None = None,
    device: torch.device | None = None,
) -> str:
    """
    Full pipeline: load audio → run Grad-CAM → save figure.

    Args:
        audio_path  : Path to the audio file to explain.
        output_path : Where to save the output PNG.
        model_path  : Override config model path.
        device      : Override compute device.

    Returns:
        Path to the saved figure.
    """
    device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_path or _cfg["app"]["model_path"]

    model       = load_model(_cfg, model_path, device)
    target_layer = get_target_layer(model)
    gradcam      = GradCAM(model, target_layer)

    lfcc = extract_lfcc(audio_path)                             # (N_LFCC, T)
    x    = torch.tensor(lfcc).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,N,T)

    heatmap = gradcam.generate(x, target_class=1)               # (H, W)

    # Upsample heatmap to match LFCC dimensions for overlay
    from scipy.ndimage import zoom
    scale   = (lfcc.shape[0] / heatmap.shape[0], lfcc.shape[1] / heatmap.shape[1])
    heatmap = zoom(heatmap, scale, order=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)

    im0 = axes[0].imshow(lfcc, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("LFCC Spectrogram", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("LFCC Coefficient")
    plt.colorbar(im0, ax=axes[0], fraction=0.02)

    im1 = axes[1].imshow(lfcc,    aspect="auto", origin="lower", cmap="magma", alpha=0.6)
    im2 = axes[1].imshow(heatmap, aspect="auto", origin="lower", cmap="hot",   alpha=0.6)
    axes[1].set_title("Grad-CAM Overlay — Brighter = More Suspicious", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("LFCC Coefficient")
    axes[1].set_xlabel("Time Frame")
    plt.colorbar(im2, ax=axes[1], fraction=0.02, label="Suspicion Score")

    fig.suptitle(
        f"Forensic Audio Analysis: {os.path.basename(audio_path)}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grad-CAM plot saved → %s", output_path)
    return output_path
