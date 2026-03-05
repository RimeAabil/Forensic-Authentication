"""
generate_dataset.py — Master script to generate N tampered audio clips.

Randomly applies one of four tampering methods to authentic + deepfake
source clips, writing labeled output to data/tampered/.

Usage:
    python tampering/generate_dataset.py
    python tampering/generate_dataset.py --num_samples 2000 --sr 16000
"""

import os
import sys
import random
import glob
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tampering.splice           import splice_audio
from tampering.crossfade        import crossfade_splice
from tampering.speed_change     import speed_tamper
from tampering.inject_synthetic import inject_synthetic

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("generate_dataset")

# ── Constants ─────────────────────────────────────────────────────────────────
AUTHENTIC_DIR = "data/authentic"
DEEPFAKES_DIR = "data/deepfakes"
OUTPUT_DIR    = "data/tampered"
METHODS       = ["splice", "crossfade", "speed", "inject"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tampered audio dataset")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--sr",          type=int, default=16000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--authentic_dir", default=AUTHENTIC_DIR)
    parser.add_argument("--deepfakes_dir", default=DEEPFAKES_DIR)
    parser.add_argument("--output_dir",    default=OUTPUT_DIR)
    return parser.parse_args()


def collect_files(directory: str) -> list:
    files = (
        glob.glob(f"{directory}/**/*.wav",  recursive=True) +
        glob.glob(f"{directory}/**/*.flac", recursive=True) +
        glob.glob(f"{directory}/**/*.mp3",  recursive=True)
    )
    return sorted(files)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    authentic = collect_files(args.authentic_dir)
    deepfakes = collect_files(args.deepfakes_dir)

    if len(authentic) < 2:
        logger.error("Need at least 2 authentic files in %s", args.authentic_dir)
        sys.exit(1)

    if len(deepfakes) < 1:
        logger.warning("No deepfake files found — 'inject' method will be skipped")
        available_methods = [m for m in METHODS if m != "inject"]
    else:
        available_methods = METHODS

    logger.info("Authentic clips : %d", len(authentic))
    logger.info("Deepfake clips  : %d", len(deepfakes))
    logger.info("Generating %d tampered clips...", args.num_samples)

    success_count = 0
    fail_count    = 0

    for i in tqdm(range(args.num_samples), desc="Generating", unit="clip"):
        method = random.choice(available_methods)
        f1     = random.choice(authentic)
        f2     = random.choice(authentic)
        output = os.path.join(args.output_dir, f"tampered_{i:05d}_{method}.wav")

        ok = False
        if method == "splice":
            ok = splice_audio(f1, f2, output, sr=args.sr)
        elif method == "crossfade":
            ok = crossfade_splice(f1, f2, output, sr=args.sr)
        elif method == "speed":
            ok = speed_tamper(f1, output, sr=args.sr)
        elif method == "inject":
            fake = random.choice(deepfakes)
            ok   = inject_synthetic(f1, fake, output, sr=args.sr)

        if ok:
            success_count += 1
        else:
            fail_count += 1

    logger.info("Done — Success: %d | Failed: %d", success_count, fail_count)
    logger.info("Output saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
