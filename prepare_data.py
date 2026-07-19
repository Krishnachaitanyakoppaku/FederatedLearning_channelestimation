"""
prepare_data.py — Data Preparation & Train-Test Split for CNN Channel Estimation

Pipeline Steps:
  1. Load the real/imaginary–split clean and noisy channel matrices.
  2. Perform an 80/20 train-test split with controlled shuffling.
  3. Save X_train, Y_train, X_test, Y_test as .npy files.

Input shapes  : (N, 2, 64, 64) float32   — [real, imag] stacked along axis-1
Output files  : dataset/cnn/X_train.npy, Y_train.npy, X_test.npy, Y_test.npy

Usage:
    python prepare_data.py --snr 10

Author : Project Team
"""

import numpy as np
import os
import argparse

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
SEED        = 42                     # Ensures reproducible shuffling
TRAIN_RATIO = 0.80                   # 80 % training, 20 % testing
DATASET_DIR = "dataset"              # Root directory of saved .npy files
OUTPUT_DIR  = "dataset/cnn"          # Where train/test splits are saved


def load_and_split(snr_db: int) -> None:
    """
    Load clean (label) and noisy (input) channels for a specific SNR,
    shuffle them with a fixed seed, split 80/20, and save to disk.

    Parameters
    ----------
    snr_db : int
        SNR level in dB (must match a file saved by add_noise.py).
    """

    # ── Step 1: Load the real/imaginary-split matrices ────────
    #   X  = noisy channel  → CNN input
    #   Y  = clean channel  → ground-truth label
    X = np.load(os.path.join(DATASET_DIR, f"H_noisy_{snr_db}dB_ri.npy"))
    Y = np.load(os.path.join(DATASET_DIR, "H_clean_ri.npy"))

    print(f"[INFO] Loaded data for SNR = {snr_db} dB")
    print(f"       X (noisy)  : {X.shape}  dtype={X.dtype}")
    print(f"       Y (clean)  : {Y.shape}  dtype={Y.dtype}")

    # ── Step 2: Global normalization ─────────────────────────────
    #   The raw channel values are ~1e-6, far too small for the CNN
    #   to learn a useful residual. We normalize to zero-mean,
    #   unit-variance using stats computed over the FULL dataset
    #   (both X and Y, all samples) so the scale is consistent.
    #   The stats are saved so evaluation can invert the transform.
    global_mean = np.mean(np.concatenate([X, Y], axis=0)).astype(np.float32)
    global_std  = np.std(np.concatenate([X, Y], axis=0)).astype(np.float32)

    X = ((X - global_mean) / global_std).astype(np.float32)
    Y = ((Y - global_mean) / global_std).astype(np.float32)

    print(f"\n[INFO] Normalization applied")
    print(f"       global_mean = {global_mean:.10f}")
    print(f"       global_std  = {global_std:.10f}")
    print(f"       X after norm : mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"       Y after norm : mean={Y.mean():.4f}, std={Y.std():.4f}")

    # ── Step 3: Shuffle both arrays in unison ─────────────────
    #   A fixed seed guarantees the same split every run.
    n_samples = X.shape[0]
    rng     = np.random.default_rng(SEED)
    indices = rng.permutation(n_samples)

    X = X[indices]
    Y = Y[indices]

    # ── Step 4: 80/20 train-test split ────────────────────────
    split_idx = int(n_samples * TRAIN_RATIO)

    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"\n[INFO] Train-Test Split")
    print(f"       Train  →  X: {X_train.shape}   Y: {Y_train.shape}")
    print(f"       Test   →  X: {X_test.shape}   Y: {Y_test.shape}")

    # ── Step 5: Save to disk ──────────────────────────────────
    save_dir = os.path.join(OUTPUT_DIR, f"{snr_db}dB")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(save_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(save_dir, "Y_test.npy"),  Y_test)

    # Save normalization stats for inverse transform during evaluation
    np.savez(os.path.join(save_dir, "norm_stats.npz"),
             mean=global_mean, std=global_std)

    print(f"\n[✓] Saved train/test splits + norm_stats → {save_dir}/")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare CNN train/test data for a specific SNR level."
    )
    parser.add_argument(
        "--snr", type=int, default=10,
        help="SNR level in dB (default: 10)"
    )
    args = parser.parse_args()

    load_and_split(args.snr)
