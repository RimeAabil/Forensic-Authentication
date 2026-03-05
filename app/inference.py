"""
inference.py — Production inference engine.

Loads the trained model once (singleton pattern) and exposes a
single predict() function used by both the Gradio app and any
external API integration.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.extract_lfcc import extract_lfcc
from model.cnn_lstm        import load_model, CNNLSTM

logger = logging.getLogger(__name__)

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Singleton model instance — loaded once on first call
_model: Optional[CNNLSTM] = None


def _get_model() -> CNNLSTM:
    global _model
    if _model is None:
        model_path = _cfg["app"]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run `python model/train.py` first."
            )
        _model = load_model(_cfg, model_path, DEVICE)
        logger.info("Model loaded onto %s", DEVICE)
    return _model


def predict(audio_path: str) -> dict:
    """
    Run inference on a single audio file.

    Args:
        audio_path: Path to a .wav / .flac / .mp3 file.

    Returns:
        dict with keys:
            authentic_prob (float) : Probability the audio is authentic [0, 1].
            tampered_prob  (float) : Probability the audio is tampered  [0, 1].
            verdict        (str)   : "AUTHENTIC" or "TAMPERED".
            confidence     (float) : Max probability (certainty of verdict).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()

    try:
        lfcc = extract_lfcc(audio_path)
    except Exception as exc:
        raise RuntimeError(f"Feature extraction failed for {audio_path}: {exc}") from exc

    x = torch.tensor(lfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    authentic_prob = float(probs[0])
    tampered_prob  = float(probs[1])
    verdict        = "TAMPERED" if tampered_prob > 0.5 else "AUTHENTIC"
    confidence     = float(max(probs))

    logger.info(
        "Prediction [%s]: %s  (authentic=%.3f  tampered=%.3f  confidence=%.3f)",
        os.path.basename(audio_path), verdict,
        authentic_prob, tampered_prob, confidence,
    )

    return {
        "authentic_prob": authentic_prob,
        "tampered_prob":  tampered_prob,
        "verdict":        verdict,
        "confidence":     confidence,
    }
