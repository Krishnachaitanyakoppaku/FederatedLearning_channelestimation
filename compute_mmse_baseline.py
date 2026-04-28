"""
Full-Covariance MMSE Baseline for Channel Estimation
=====================================================
Computes the Linear MMSE (Wiener filter) estimate:
    H_hat = R_HH (R_HH + sigma^2 I)^{-1} H_noisy

where R_HH is the sample covariance of the clean channel vectors,
and sigma^2 is derived from the SNR level.

This replaces the scalar-calibrated LMMSE with a proper full-covariance MMSE.
"""

import numpy as np
import time

SNR_LEVELS = [0, 5, 10, 15, 20]

def nmse_db(h_true, h_est):
    """Compute NMSE in dB across the batch."""
    num = np.sum((h_true - h_est) ** 2, axis=(1, 2, 3))
    den = np.sum(h_true ** 2, axis=(1, 2, 3))
    nmse_linear = num / den
    return 10.0 * np.log10(np.mean(nmse_linear))

def compute_mmse_for_snr(snr_db):
    """Compute full-covariance MMSE for a given SNR level."""
    base = f"dataset/cnn/{snr_db}dB"
    
    X_train = np.load(f"{base}/X_train.npy")  # noisy (N, 2, 64, 64)
    Y_train = np.load(f"{base}/Y_train.npy")  # clean  (N, 2, 64, 64)
    X_test  = np.load(f"{base}/X_test.npy")
    Y_test  = np.load(f"{base}/Y_test.npy")
    
    N_train = Y_train.shape[0]
    N_test  = X_test.shape[0]
    
    # Flatten each sample: (N, 2, 64, 64) -> (N, 2*64*64) = (N, 8192)
    d = 2 * 64 * 64  # 8192
    H_train_flat = Y_train.reshape(N_train, d)
    X_test_flat  = X_test.reshape(N_test, d)
    Y_test_flat  = Y_test.reshape(N_test, d)
    
    # Estimate noise variance from training data
    noise_train = X_train.reshape(N_train, d) - H_train_flat
    sigma2 = np.mean(noise_train ** 2)
    
    # Compute sample covariance R_HH = (1/N) H^T H
    # For memory efficiency, compute in float64
    H_train_f64 = H_train_flat.astype(np.float64)
    mean_h = np.mean(H_train_f64, axis=0, keepdims=True)
    H_centered = H_train_f64 - mean_h
    R_HH = (H_centered.T @ H_centered) / N_train  # (d, d)
    
    print(f"  R_HH shape: {R_HH.shape}, sigma^2: {sigma2:.6f}")
    
    # MMSE filter: W = R_HH (R_HH + sigma^2 I)^{-1}
    I = np.eye(d, dtype=np.float64)
    t0 = time.time()
    W = R_HH @ np.linalg.inv(R_HH + sigma2 * I)  # (d, d)
    t_inv = time.time() - t0
    print(f"  Matrix inversion took {t_inv:.2f}s")
    
    # Apply MMSE to test data
    X_test_f64 = X_test_flat.astype(np.float64)
    X_centered = X_test_f64 - mean_h  # center using training mean
    H_est_centered = (W @ X_centered.T).T  # (N_test, d)
    H_est_flat = H_est_centered + mean_h
    
    # Reshape back to (N, 2, 64, 64)
    H_est = H_est_flat.reshape(N_test, 2, 64, 64).astype(np.float32)
    
    # Compute NMSE
    result = nmse_db(Y_test, H_est)
    
    # Also compute scalar LMMSE for comparison
    # Scalar: h_hat = (P_h / (P_h + sigma2)) * x
    P_h = np.mean(H_train_flat ** 2)
    alpha = P_h / (P_h + sigma2)
    H_est_scalar = alpha * X_test_flat
    scalar_result = nmse_db(Y_test_flat.reshape(N_test, 2, 64, 64),
                            H_est_scalar.reshape(N_test, 2, 64, 64))
    
    # LS baseline (just the noisy observation)
    ls_result = nmse_db(Y_test, X_test)
    
    return result, scalar_result, ls_result

if __name__ == "__main__":
    print("=" * 70)
    print("Full-Covariance MMSE Baseline Computation")
    print("=" * 70)
    
    results = {}
    for snr in SNR_LEVELS:
        print(f"\n--- SNR = {snr} dB ---")
        try:
            mmse, scalar, ls = compute_mmse_for_snr(snr)
            results[snr] = (mmse, scalar, ls)
            print(f"  LS (raw noisy):       {ls:.4f} dB")
            print(f"  Scalar LMMSE:         {scalar:.4f} dB")
            print(f"  Full-Covariance MMSE: {mmse:.4f} dB")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'SNR (dB)':>10} | {'LS (dB)':>12} | {'Scalar LMMSE':>14} | {'Full-Cov MMSE':>15}")
    print("-" * 60)
    for snr in sorted(results.keys()):
        mmse, scalar, ls = results[snr]
        print(f"{snr:>10} | {ls:>12.4f} | {scalar:>14.4f} | {mmse:>15.4f}")
