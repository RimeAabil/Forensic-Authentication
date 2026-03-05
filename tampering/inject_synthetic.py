"""
inject_synthetic.py — Insert a deepfake/synthetic segment into a real recording.

Simulates a forensic scenario where an attacker replaces a word or phrase
in an authentic recording with an AI-generated substitute.
"""

import logging
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def inject_synthetic(
    real_file: str,
    fake_file: str,
    output_path: str,
    sr: int = 16000,
    inject_duration_s: float = 1.0,
    position_range: tuple = (0.2, 0.6),
) -> bool:
    """
    Replace a segment of a real audio clip with synthetic/deepfake audio.

    Args:
        real_file         : Path to the authentic source audio.
        fake_file         : Path to the synthetic/deepfake audio.
        output_path       : Destination path for the injected output.
        sr                : Target sample rate.
        inject_duration_s : Duration in seconds of the injected segment.
        position_range    : (min, max) fraction range for injection start point.

    Returns:
        True on success, False on failure.
    """
    try:
        real, _ = librosa.load(real_file, sr=sr, mono=True)
        fake, _ = librosa.load(fake_file, sr=sr, mono=True)

        inject_len = int(sr * inject_duration_s)

        if len(real) < inject_len * 2:
            logger.warning("Real clip too short for injection: %s", real_file)
            return False

        if len(fake) == 0:
            logger.warning("Fake clip is empty: %s", fake_file)
            return False

        # Trim or loop the fake segment to exactly inject_len samples
        if len(fake) >= inject_len:
            fake_segment = fake[:inject_len]
        else:
            repeats      = int(np.ceil(inject_len / len(fake)))
            fake_segment = np.tile(fake, repeats)[:inject_len]

        inject_start = int(len(real) * np.random.uniform(*position_range))
        inject_end   = inject_start + inject_len

        # Clamp to audio boundaries
        inject_end = min(inject_end, len(real))
        fake_segment = fake_segment[: inject_end - inject_start]

        result = np.concatenate([
            real[:inject_start],
            fake_segment,
            real[inject_end:]
        ])

        sf.write(output_path, result, sr)
        logger.debug("Synthetic injection written to %s", output_path)
        return True

    except Exception as exc:
        logger.error("inject_synthetic failed [%s | %s]: %s", real_file, fake_file, exc)
        return False
