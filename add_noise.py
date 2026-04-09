"""
add_noise.py — AWGN Channel Corruption for DeepMIMO CSI Dataset

Adds circularly symmetric complex Gaussian (CSCG) noise at multiple
SNR levels using per-sample power normalization. Outputs both complex
and real/imaginary-split formats.

"""

import numpy as np
import os

# --- Configuration ---
SEED       = 42
SNR_DB     = [0, 5, 10, 15, 20]
OUTPUT_DIR = "dataset"

# --- Reproducible RNG ---
rng = np.random.default_rng(SEED)


def add_awgn(H, snr_db, rng):
    """Apply per-sample AWGN at the given SNR (dB).

    Noise variance is set independently for each user based on its
    own channel power, ensuring the target SNR holds per-sample —
    critical for mmWave scenarios with large LOS/NLOS power variation.
    """
    snr_lin   = 10 ** (snr_db / 10)
    sig_power = np.mean(np.abs(H) ** 2, axis=(1, 2))                   # (N,)
    std       = np.sqrt(sig_power / (2 * snr_lin))[:, None, None]       # (N,1,1)
    noise     = (rng.standard_normal(H.shape)
                 + 1j * rng.standard_normal(H.shape)) * std
    return H + noise


def to_real_imag(H):
    """Stack real and imaginary parts: (N,Nt,Ns) → (N,2,Nt,Ns) float32."""
    return np.stack((H.real, H.imag), axis=1).astype(np.float32)


# --- Load & prepare ---
H = np.load("o1_60_matrix.npy").astype(np.complex64)    # force complex64
H = H.squeeze(axis=1)                                   # (N,1,64,64) → (N,64,64)
print(f"Loaded: {H.shape}  dtype={H.dtype}  users={H.shape[0]}")

# --- Save clean reference ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(f"{OUTPUT_DIR}/H_clean.npy",    H)
np.save(f"{OUTPUT_DIR}/H_clean_ri.npy", to_real_imag(H))
print(f"Saved clean channels → {OUTPUT_DIR}/")

# --- Generate noisy versions at each SNR ---
for snr in SNR_DB:
    H_n = add_awgn(H, snr, rng)

    np.save(f"{OUTPUT_DIR}/H_noisy_{snr}dB.npy",    H_n)
    np.save(f"{OUTPUT_DIR}/H_noisy_{snr}dB_ri.npy", to_real_imag(H_n))

    # Verify achieved SNR
    actual = 10 * np.log10(np.mean(np.abs(H)**2) / np.mean(np.abs(H_n - H)**2))
    print(f"  {snr:>2d} dB  →  actual {actual:.2f} dB  |  {H_n.shape} complex64")

print(f"\n Dataset saved to '{OUTPUT_DIR}/'  (seed={SEED})")