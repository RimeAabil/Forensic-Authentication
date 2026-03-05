"""
splice.py — Hard-cut splicing between two audio clips.

Simulates a forensic scenario where an editor removes or replaces a segment
by cutting one clip and joining it with another at a random point.
"""

import logging
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def splice_audio(
    file1: str,
    file2: str,
    output_path: str,
    sr: int = 16000,
    cut_range: tuple = (0.3, 0.7),
) -> bool:
    """
    Load two audio files and splice them at a random cut point.

    Args:
        file1       : Path to the first (primary) audio file.
        file2       : Path to the second audio file used for the tail segment.
        output_path : Destination path for the spliced output.
        sr          : Target sample rate.
        cut_range   : Tuple (min, max) fraction range for the cut point.

    Returns:
        True on success, False on failure.
    """
    try:
        audio1, _ = librosa.load(file1, sr=sr, mono=True)
        audio2, _ = librosa.load(file2, sr=sr, mono=True)

        if len(audio1) == 0 or len(audio2) == 0:
            logger.warning("One of the input files is empty: %s | %s", file1, file2)
            return False

        # Align lengths to the shorter clip to avoid index overflow
        min_len = min(len(audio1), len(audio2))
        audio1  = audio1[:min_len]
        audio2  = audio2[:min_len]

        cut = int(min_len * np.random.uniform(*cut_range))
        spliced = np.concatenate([audio1[:cut], audio2[cut:]])

        sf.write(output_path, spliced, sr)
        logger.debug("Splice written to %s (cut@%d/%d)", output_path, cut, min_len)
        return True

    except Exception as exc:
        logger.error("splice_audio failed [%s | %s]: %s", file1, file2, exc)
        return False
