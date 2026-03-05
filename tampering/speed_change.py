"""
speed_change.py — Time-stretch a segment of an audio clip.

Simulates forensic tampering where a forger speeds up or slows down
a portion of a recording to alter perceived timing or hide deletions.
"""

import logging
import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def speed_tamper(
    file1: str,
    output_path: str,
    sr: int = 16000,
    rate_range: tuple = (0.85, 1.15),
    segment_range: tuple = (0.25, 0.75),
) -> bool:
    """
    Time-stretch a random mid-segment of the audio.

    Args:
        file1          : Path to the source audio file.
        output_path    : Destination path for the tampered output.
        sr             : Target sample rate.
        rate_range     : (min, max) stretch factor — values < 1 slow down, > 1 speed up.
        segment_range  : (start_frac, end_frac) for the segment to tamper.

    Returns:
        True on success, False on failure.
    """
    try:
        audio, _ = librosa.load(file1, sr=sr, mono=True)

        if len(audio) < sr * 0.5:
            logger.warning("Clip too short for speed tampering: %s", file1)
            return False

        rate  = float(np.random.uniform(*rate_range))
        start = int(len(audio) * segment_range[0])
        end   = int(len(audio) * segment_range[1])

        segment  = audio[start:end]
        stretched = librosa.effects.time_stretch(segment, rate=rate)
        tampered  = np.concatenate([audio[:start], stretched, audio[end:]])

        sf.write(output_path, tampered, sr)
        logger.debug("Speed tamper written to %s (rate=%.3f)", output_path, rate)
        return True

    except Exception as exc:
        logger.error("speed_tamper failed [%s]: %s", file1, exc)
        return False
