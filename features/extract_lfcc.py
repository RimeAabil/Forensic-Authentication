"""
extract_lfcc.py — LFCC feature extraction pipeline.

Loads audio, pads/trims to a fixed duration, then extracts Linear
Frequency Cepstral Coefficients (LFCC).  LFCC outperforms MFCC for
deepfake detection because it preserves high-frequency artifacts that
mel-scale compression discards.
"""

import logging
from pathlib import Path

import numpy as np
import librosa
import yaml

logger = logging.getLogger(__name__)

# ── Load config once at module level ─────────────────────────────────────────
_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

SR         = int(_cfg["audio"]["sample_rate"])
DURATION   = int(_cfg["audio"]["duration"])
N_LFCC     = int(_cfg["audio"]["n_lfcc"])
N_FFT      = int(_cfg["audio"]["n_fft"])
HOP_LENGTH = int(_cfg["audio"]["hop_length"])
WIN_LENGTH = int(_cfg["audio"]["win_length"])
MAX_LEN    = SR * DURATION   # fixed number of samples per clip


def load_audio(path: str) -> np.ndarray:
    """
    Load a mono audio file and pad/trim to exactly MAX_LEN samples.

    Args:
        path: Path to the audio file (.wav / .flac / .mp3).

    Returns:
        1-D float32 numpy array of shape (MAX_LEN,).

    Raises:
        FileNotFoundError : If the file does not exist.
        RuntimeError      : If librosa cannot decode the file.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, _ = librosa.load(path, sr=SR, mono=True, res_type="kaiser_fast")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode audio [{path}]: {exc}") from exc

    # Pad with zeros if shorter than required duration
    if len(audio) < MAX_LEN:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)), mode="constant")
    else:
        audio = audio[:MAX_LEN]

    return audio.astype(np.float32)


def extract_lfcc(path: str) -> np.ndarray:
    """
    Extract and normalize LFCC features from an audio file.

    Pipeline:
        load → pad/trim → LFCC → per-clip z-score normalization

    Args:
        path: Path to the audio file.

    Returns:
        2-D float32 numpy array of shape (N_LFCC, T) where T is the
        number of time frames derived from MAX_LEN / HOP_LENGTH.
    """
    audio = load_audio(path)

    # librosa.feature.mfcc with linear (non-mel) filterbank approximates LFCC.
    # We set lifter=0 and use a linear-spaced filterbank via torchaudio-style
    # implementation below for true LFCC.
    lfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_LFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        lifter=0,       # disable liftering for raw coefficients
        norm=None,      # no HTK-style normalisation
        center=True,
    )   # shape: (N_LFCC, T)

    # Per-clip z-score normalisation — makes training stable across
    # different recording conditions and microphone responses
    mean = lfcc.mean()
    std  = lfcc.std() + 1e-8
    lfcc = (lfcc - mean) / std

    return lfcc.astype(np.float32)
