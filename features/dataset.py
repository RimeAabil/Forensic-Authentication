"""
dataset.py — PyTorch Dataset for LFCC audio features.

Wraps a CSV manifest (path, label) and exposes (lfcc_tensor, label_tensor)
pairs for DataLoader consumption.  Includes robust error handling so a
single corrupt file never crashes an entire training epoch.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Resolve features directory for relative imports regardless of CWD
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_lfcc import extract_lfcc, N_LFCC

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """
    PyTorch Dataset that loads audio files on-the-fly and returns
    pre-computed LFCC feature tensors.

    Args:
        csv_path    : Path to a CSV file with columns [path, label].
        augment     : If True, apply basic time-domain augmentation
                      (only used during training to improve robustness).

    Returns per item:
        lfcc  : Float32 tensor of shape (1, N_LFCC, T) — channel-first for CNN.
        label : Long tensor scalar (0=authentic, 1=tampered).
    """

    def __init__(self, csv_path: str, augment: bool = False) -> None:
        self.df      = pd.read_csv(csv_path)
        self.augment = augment
        self._validate()
        logger.info("Dataset loaded: %d samples from %s", len(self.df), csv_path)

    def _validate(self) -> None:
        required = {"path", "label"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        original  = len(self.df)
        self.df   = self.df.dropna(subset=["path", "label"]).reset_index(drop=True)
        dropped   = original - len(self.df)
        if dropped:
            logger.warning("Dropped %d rows with NaN values", dropped)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        try:
            lfcc = extract_lfcc(str(row["path"]))  # (N_LFCC, T)
        except Exception as exc:
            logger.error("Failed to extract LFCC for %s: %s", row["path"], exc)
            # Return a zero tensor so collate_fn can still batch this item
            # without crashing — the model will learn to ignore zero-valued inputs
            lfcc = np.zeros((N_LFCC, 401), dtype=np.float32)

        if self.augment:
            lfcc = self._augment(lfcc)

        lfcc_tensor  = torch.tensor(lfcc, dtype=torch.float32).unsqueeze(0)  # (1, N_LFCC, T)
        label_tensor = torch.tensor(int(row["label"]), dtype=torch.long)
        return lfcc_tensor, label_tensor

    @staticmethod
    def _augment(lfcc: np.ndarray) -> np.ndarray:
        """
        Lightweight augmentation applied in feature space.
        - Random Gaussian noise injection (SNR ~30 dB)
        - Random frequency masking (SpecAugment-style)
        - Random time masking
        """
        lfcc = lfcc.copy()

        # Gaussian noise
        if np.random.rand() < 0.5:
            noise = np.random.randn(*lfcc.shape).astype(np.float32) * 0.05
            lfcc += noise

        # Frequency masking: zero out up to 20% of LFCC rows
        if np.random.rand() < 0.5:
            n_mask = int(lfcc.shape[0] * 0.2)
            f0     = np.random.randint(0, lfcc.shape[0] - n_mask)
            lfcc[f0 : f0 + n_mask, :] = 0.0

        # Time masking: zero out up to 10% of time frames
        if np.random.rand() < 0.5:
            n_mask = int(lfcc.shape[1] * 0.1)
            t0     = np.random.randint(0, lfcc.shape[1] - n_mask)
            lfcc[:, t0 : t0 + n_mask] = 0.0

        return lfcc
