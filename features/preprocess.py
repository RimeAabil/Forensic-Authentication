"""
preprocess.py — Build file list and create train/val/test CSV splits.

Scans all audio directories, assigns labels, shuffles, stratifies,
and writes three CSV files used by the PyTorch Dataset loader.

Labels:
    0 — Authentic (genuine, unmodified speech)
    1 — Tampered  (spliced, speed-changed, injected, or deepfake)

Usage:
    python features/preprocess.py
    python features/preprocess.py --config path/to/config.yaml
"""

import os
import sys
import glob
import logging
import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset splits")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "config.yaml"),
    )
    return parser.parse_args()


def collect_files(directory: str, label: int) -> list[dict]:
    """Recursively collect audio files from a directory with a given label."""
    if not os.path.isdir(directory):
        logger.warning("Directory not found, skipping: %s", directory)
        return []

    patterns = ["**/*.wav", "**/*.flac", "**/*.mp3"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))

    rows = [{"path": os.path.abspath(f), "label": label} for f in sorted(files)]
    logger.info("Found %d files in %s (label=%d)", len(rows), directory, label)
    return rows


def validate_files(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the audio file no longer exists on disk."""
    original = len(df)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    removed = original - len(df)
    if removed > 0:
        logger.warning("Removed %d missing files from manifest", removed)
    return df


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    authentic_dir = cfg["data"]["authentic_dir"]
    tampered_dir  = cfg["data"]["tampered_dir"]
    deepfakes_dir = cfg["data"]["deepfakes_dir"]
    splits_dir    = cfg["data"]["splits_dir"]
    train_ratio   = cfg["data"]["train_ratio"]
    val_ratio     = cfg["data"]["val_ratio"]

    os.makedirs(splits_dir, exist_ok=True)

    # Collect all files with labels
    rows = (
        collect_files(authentic_dir, label=0) +
        collect_files(tampered_dir,  label=1) +
        collect_files(deepfakes_dir, label=1)
    )

    if not rows:
        logger.error("No audio files found. Check your data directories.")
        sys.exit(1)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df = validate_files(df)

    logger.info("Total samples — Authentic: %d | Tampered: %d",
                (df["label"] == 0).sum(), (df["label"] == 1).sum())

    # Stratified splits
    test_size = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )

    val_frac = val_ratio / test_size
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    # Save splits
    train_path = os.path.join(splits_dir, "train.csv")
    val_path   = os.path.join(splits_dir, "val.csv")
    test_path  = os.path.join(splits_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    logger.info("Splits saved to %s", splits_dir)
    logger.info("  Train : %d samples", len(train_df))
    logger.info("  Val   : %d samples", len(val_df))
    logger.info("  Test  : %d samples", len(test_df))


if __name__ == "__main__":
    main()
