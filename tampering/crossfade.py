"""
crossfade.py — Smooth crossfade splicing to hide edit boundaries.

Applies a linear fade-out / fade-in envelope at the join point, producing
a perceptually smoother transition that is harder to detect by ear but
still leaves spectral phase artifacts detectable by ML.
"""

import logging
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def crossfade_splice(
    file1: str,
    file2: str,
    output_path: str,
    sr: int = 16000,
    fade_ms: float = 20.0,
    cut_range: tuple = (0.3, 0.7),
) -> bool:
    """
    Splice two audio files with a linear crossfade at the join point.

    Args:
        file1       : Path to the primary audio file.
        file2       : Path to the secondary audio file.
        output_path : Destination path for the output.
        sr          : Target sample rate.
        fade_ms     : Duration of the crossfade in milliseconds.
        cut_range   : Fraction range for the cut point.

    Returns:
        True on success, False on failure.
    """
    try:
        audio1, _ = librosa.load(file1, sr=sr, mono=True)
        audio2, _ = librosa.load(file2, sr=sr, mono=True)

        if len(audio1) == 0 or len(audio2) == 0:
            logger.warning("Empty input: %s | %s", file1, file2)
            return False

        min_len      = min(len(audio1), len(audio2))
        audio1       = audio1[:min_len].copy()
        audio2       = audio2[:min_len].copy()
        fade_samples = int(sr * fade_ms / 1000)
        cut          = int(min_len * np.random.uniform(*cut_range))

        # Guard: ensure enough samples on both sides for the fade
        if cut < fade_samples or (min_len - cut) < fade_samples:
            logger.warning("Clip too short for requested fade — skipping fade")
            sf.write(output_path, np.concatenate([audio1[:cut], audio2[cut:]]), sr)
            return True

        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        fade_in  = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        # Apply envelopes at the join point
        audio1[cut - fade_samples : cut] *= fade_out
        audio2[cut : cut + fade_samples] *= fade_in

        result = np.concatenate([audio1[:cut], audio2[cut:]])
        sf.write(output_path, result, sr)
        logger.debug("Crossfade written to %s", output_path)
        return True

    except Exception as exc:
        logger.error("crossfade_splice failed [%s | %s]: %s", file1, file2, exc)
        return False
