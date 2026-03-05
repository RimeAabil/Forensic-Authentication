"""
cnn_lstm.py — CNN-BiLSTM hybrid model for forensic audio authentication.

Architecture:
    Input  : (B, 1, N_LFCC, T)  — batch of LFCC spectrograms
    CNN    : 3 × Conv2d blocks → extracts local spectral patterns
    BiLSTM : 2-layer bidirectional LSTM → captures temporal inconsistencies
    Head   : Linear → 2-class softmax (authentic / tampered)

Design rationale:
    - CNN sees the LFCC matrix as a 2-D image; MaxPool reduces spatial dims.
    - AdaptiveAvgPool collapses the frequency axis to 1 so the temporal axis
      is fed directly into the LSTM regardless of input length.
    - Bidirectional LSTM reads the sequence both forwards and backwards —
      critical for detecting edit boundaries that are only obvious in context.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool with optional Dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        pool: tuple[int, int] = (2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNLSTM(nn.Module):
    """
    CNN-BiLSTM forensic audio classifier.

    Args:
        cfg: Parsed config dict (loaded from config.yaml).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        filters     = cfg["model"]["cnn_filters"]    # e.g. [32, 64, 128]
        lstm_hidden = cfg["model"]["lstm_hidden"]    # 128
        lstm_layers = cfg["model"]["lstm_layers"]    # 2
        dropout     = cfg["model"]["dropout"]        # 0.3
        num_classes = cfg["model"]["num_classes"]    # 2

        # ── CNN backbone ──────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            ConvBlock(1,          filters[0], pool=(2, 2), dropout=dropout * 0.5),
            ConvBlock(filters[0], filters[1], pool=(2, 2), dropout=dropout * 0.5),
            ConvBlock(filters[1], filters[2], pool=(2, 1), dropout=dropout * 0.5),
            # Collapse frequency axis → shape becomes (B, filters[2], 1, T')
            nn.AdaptiveAvgPool2d((1, None)),
        )

        # ── Bidirectional LSTM ────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=filters[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Attention-weighted pooling ────────────────────────────────────
        # Instead of just taking the last hidden state, we learn a soft
        # attention mask over all time steps — more robust for long clips.
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()
        logger.info("CNNLSTM built — params: %s", f"{self._count_params():,}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, N_LFCC, T).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # CNN: (B, 1, F, T) → (B, C, 1, T')
        x = self.cnn(x)

        # Reshape for LSTM: (B, C, 1, T') → (B, T', C)
        B, C, _, T = x.shape
        x = x.squeeze(2).permute(0, 2, 1)   # (B, T', C)

        # BiLSTM: (B, T', C) → (B, T', H*2)
        x, _ = self.lstm(x)

        # Attention pooling over time steps
        attn_weights = torch.softmax(self.attention(x), dim=1)   # (B, T', 1)
        x = (x * attn_weights).sum(dim=1)                        # (B, H*2)

        return self.classifier(x)   # (B, num_classes)


def load_model(cfg: dict, weights_path: str, device: torch.device) -> CNNLSTM:
    """
    Convenience function: instantiate CNNLSTM and load saved weights.

    Args:
        cfg          : Config dict.
        weights_path : Path to a saved .pth state dict.
        device       : torch.device to map weights to.

    Returns:
        Loaded model in eval mode.
    """
    model = CNNLSTM(cfg).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded from %s", weights_path)
    return model
