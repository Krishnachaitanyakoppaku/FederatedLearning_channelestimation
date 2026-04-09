import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


# ══════════════════════════════════════════════════════════════
#  1.  CUSTOM PYTORCH DATASET
# ══════════════════════════════════════════════════════════════

class ChannelDataset(Dataset):
    """
    PyTorch Dataset that wraps pre-saved .npy train/test arrays.

    Parameters
    ----------
    x_path : str
        Path to the noisy channel matrix  (N, 2, 64, 64)  float32.
    y_path : str
        Path to the clean  channel matrix  (N, 2, 64, 64)  float32.
    """

    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path)       # (N, 2, 64, 64)
        self.Y = np.load(y_path)       # (N, 2, 64, 64)
        assert self.X.shape == self.Y.shape, \
            f"Shape mismatch: X={self.X.shape}  Y={self.Y.shape}"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Convert numpy arrays to PyTorch tensors on access
        x = torch.from_numpy(self.X[idx])   # (2, 64, 64) float32
        y = torch.from_numpy(self.Y[idx])   # (2, 64, 64) float32
        return x, y


# ══════════════════════════════════════════════════════════════
#  2.  RESIDUAL BLOCK
# ══════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    A single residual block:  Conv → BN → ReLU → Conv → BN  + skip.

    Keeps spatial dimensions and channel count unchanged so the skip
    connection is a simple identity (no projection needed).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)      # F(x) + x


# ══════════════════════════════════════════════════════════════
#  3.  CNN CHANNEL ESTIMATOR
# ══════════════════════════════════════════════════════════════

class ChannelEstimatorCNN(nn.Module):
    """
    Lightweight residual CNN for denoising MIMO-OFDM channels.

    The model predicts a *residual correction* that is added to
    the noisy input  →  H_est = H_noisy + CNN(H_noisy).
    This "residual learning" strategy lets the network focus only
    on the noise component, converging faster and more accurately.
    """

    def __init__(self):
        super().__init__()

        # ── Feature extraction head ──────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Residual refinement body ─────────────────────────
        self.body = ResidualBlock(32)

        # ── Reconstruction tail ──────────────────────────────
        #   Projects the 32-channel feature maps back to 2
        #   channels (real + imaginary).
        self.tail = nn.Conv2d(32, 2, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (B, 2, 64, 64)
            Noisy channel (real/imag stacked).

        Returns
        -------
        Tensor, shape (B, 2, 64, 64)
            Estimated clean channel.
        """
        residual = self.tail(self.body(self.head(x)))   # learned correction
        return x + residual                              # H_est = H_noisy + Δ


# ══════════════════════════════════════════════════════════════
#  4.  UTILITY — Parameter Count
# ══════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ChannelEstimatorCNN()
    n_params = count_parameters(model)
    print(f"Model parameters : {n_params:,}")
    assert n_params < 100_000, f"Model too large: {n_params:,} params"

    # Dummy forward pass
    dummy = torch.randn(4, 2, 64, 64)
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print("[✓] Model sanity check passed.")
