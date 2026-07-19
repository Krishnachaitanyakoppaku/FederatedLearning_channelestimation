"""
train_and_evaluate.py — Training Loop & Full Evaluation for CNN Channel Estimator

Pipeline:
  1. Build PyTorch DataLoaders from the pre-split .npy files.
  2. Train the lightweight CNN using MSE loss + Adam optimizer.
  3. Evaluate on the held-out test set with multiple metrics.
  4. Support training & evaluating across ALL SNR levels.

Evaluation Metrics:
  - NMSE  (Normalized Mean Squared Error) in dB
  - MSE   (Mean Squared Error)
  - RMSE  (Root Mean Squared Error)
  - MAE   (Mean Absolute Error)
  - Cosine Similarity  (average per-sample)
  - Correlation Coefficient  (Pearson, average per-sample)

Usage:
    # Single SNR:
    python train_and_evaluate.py --snr 10 --epochs 20 --lr 0.001 --batch 32

    # All SNRs (train + evaluate each, then print summary):
    python train_and_evaluate.py --all --epochs 20 --lr 0.001 --batch 32

Dependencies : torch, numpy, cnn_model (local module)

Author : Project Team
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Local imports — model & dataset defined in cnn_model.py
from cnn_model import ChannelEstimatorCNN, ChannelDataset, count_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from research_core.logging_utils import ExperimentLogger, RunMetadata
from research_core.logging_utils import (
    detect_torch_version,
    file_sha256,
    get_git_commit,
    runtime_platform,
    runtime_python_version,
)
from research_core.metrics import nmse


# ══════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════

SNR_LEVELS = [0, 5, 10, 15, 20]       # All available SNR levels (dB)


# ══════════════════════════════════════════════════════════════
#  1.  CONFIGURATION & ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train & evaluate CNN channel estimator."
    )
    parser.add_argument("--snr",    type=int,   default=10,    help="SNR level in dB (ignored if --all)")
    parser.add_argument("--epochs", type=int,   default=20,    help="Training epochs")
    parser.add_argument("--lr",     type=float, default=1e-3,  help="Learning rate")
    parser.add_argument("--batch",  type=int,   default=32,    help="Batch size")
    parser.add_argument("--all",    action="store_true",
                        help="Train & evaluate on ALL SNR levels, then print summary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_id", type=str, default="", help="Run identifier for structured logs")
    parser.add_argument("--log_dir", type=str, default=str(PROJECT_ROOT / "results" / "raw"), help="Directory for structured logs")
    parser.add_argument("--config_path", type=str, default="", help="Optional config path for reproducibility logging")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
#  2.  DATALOADER CONSTRUCTION
# ══════════════════════════════════════════════════════════════

def build_dataloaders(snr_db: int, batch_size: int):
    """
    Wrap the saved .npy splits in PyTorch DataLoaders.

    Parameters
    ----------
    snr_db     : int   — SNR level (selects the correct sub-folder)
    batch_size : int   — mini-batch size for SGD

    Returns
    -------
    train_loader, test_loader, norm_stats : DataLoader pair + normalization dict
    """
    data_dir = f"../../dataset/cnn/{snr_db}dB"

    train_ds = ChannelDataset(
        x_path=os.path.join(data_dir, "X_train.npy"),
        y_path=os.path.join(data_dir, "Y_train.npy"),
    )
    test_ds = ChannelDataset(
        x_path=os.path.join(data_dir, "X_test.npy"),
        y_path=os.path.join(data_dir, "Y_test.npy"),
    )

    # Load normalization stats saved by prepare_data.py
    stats = np.load(os.path.join(data_dir, "norm_stats.npz"))
    norm_stats = {"mean": float(stats["mean"]), "std": float(stats["std"])}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"[INFO] DataLoaders ready  "
          f"(train={len(train_ds)}, test={len(test_ds)}, batch={batch_size})")
    print(f"[INFO] Norm stats — mean={norm_stats['mean']:.10f}, "
          f"std={norm_stats['std']:.10f}")
    return train_loader, test_loader, norm_stats


# ══════════════════════════════════════════════════════════════
#  3.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train(model, loader, criterion, optimizer, device, epoch, n_epochs):
    """
    Run one full epoch of training.

    Parameters
    ----------
    model     : nn.Module      — the CNN estimator
    loader    : DataLoader     — training data
    criterion : loss function  — MSELoss
    optimizer : optimizer      — Adam
    device    : torch.device
    epoch     : int            — current epoch (1-indexed, for display)
    n_epochs  : int            — total epochs     (for display)

    Returns
    -------
    avg_loss : float — mean MSE over all mini-batches
    """
    model.train()                        # enable dropout / BN training mode
    running_loss = 0.0

    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        x_batch = x_batch.to(device)     # (B, 2, 64, 64)
        y_batch = y_batch.to(device)     # (B, 2, 64, 64)

        # ── Forward pass ─────────────────────────────────────
        y_pred = model(x_batch)
        loss   = criterion(y_pred, y_batch)

        # ── Backward pass + weight update ────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"  Epoch [{epoch:>2d}/{n_epochs}]   MSE Loss = {avg_loss:.6f}")
    return avg_loss


# ══════════════════════════════════════════════════════════════
#  4.  EVALUATION — COMPREHENSIVE METRICS
# ══════════════════════════════════════════════════════════════

def evaluate(model, loader, device, norm_stats):
    """
    Predict on the full test set and compute comprehensive metrics.

    All metrics are computed in the ORIGINAL (un-normalized) scale
    so they are physically meaningful.

    Metrics returned
    ----------------
    nmse_db  : Normalized MSE in dB   = 10·log10( ||H−Ĥ||² / ||H||² )
    mse      : Mean Squared Error      = mean(|H−Ĥ|²)
    rmse     : Root MSE                = sqrt(MSE)
    mae      : Mean Absolute Error     = mean(|H−Ĥ|)
    cos_sim  : Cosine Similarity       (avg per-sample, on flattened complex vectors)
    pearson  : Pearson Correlation     (avg per-sample, on flattened real representation)

    Returns
    -------
    metrics : dict  — all computed metrics
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_pred  = model(x_batch)

            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(y_batch.numpy())

    # ── Concatenate all batches ───────────────────────────────
    Y_pred = np.concatenate(all_preds,  axis=0)   # (N, 2, 64, 64)
    Y_true = np.concatenate(all_labels, axis=0)   # (N, 2, 64, 64)

    # ── Un-normalize back to original scale ───────────────────
    mean = norm_stats["mean"]
    std  = norm_stats["std"]
    Y_pred = Y_pred * std + mean
    Y_true = Y_true * std + mean

    # ── Convert to complex form ──────────────────────────────
    H_pred = Y_pred[:, 0, :, :] + 1j * Y_pred[:, 1, :, :]   # (N, 64, 64)
    H_true = Y_true[:, 0, :, :] + 1j * Y_true[:, 1, :, :]   # (N, 64, 64)

    N = H_true.shape[0]

    nmse_metrics = nmse(H_true, H_pred)
    nmse_db = nmse_metrics["nmse_db"]

    # ── 2. MSE ───────────────────────────────────────────────
    mse = np.mean(np.abs(H_true - H_pred) ** 2)

    # ── 3. RMSE ──────────────────────────────────────────────
    rmse = np.sqrt(mse)

    # ── 4. MAE ───────────────────────────────────────────────
    mae = np.mean(np.abs(H_true - H_pred))

    # ── 5. Cosine Similarity (per-sample, averaged) ──────────
    #   Flatten each sample to a 1-D complex vector, compute
    #   cos_sim = Re(<h_true, h_pred>) / (||h_true|| · ||h_pred||)
    cos_sims = []
    for i in range(N):
        ht = H_true[i].flatten()
        hp = H_pred[i].flatten()
        dot     = np.real(np.vdot(ht, hp))          # Re(<ht, hp>)
        norm_t  = np.linalg.norm(ht)
        norm_p  = np.linalg.norm(hp)
        if norm_t > 0 and norm_p > 0:
            cos_sims.append(dot / (norm_t * norm_p))
    cos_sim = float(np.mean(cos_sims))

    # ── 6. Pearson Correlation (per-sample, averaged) ────────
    #   Computed on the real-valued (2, 64, 64) representation.
    pearsons = []
    for i in range(N):
        yt = Y_true[i].flatten()
        yp = Y_pred[i].flatten()
        yt_c = yt - yt.mean()
        yp_c = yp - yp.mean()
        num = np.dot(yt_c, yp_c)
        den = np.linalg.norm(yt_c) * np.linalg.norm(yp_c)
        if den > 0:
            pearsons.append(num / den)
    pearson = float(np.mean(pearsons))

    metrics = {
        "nmse_db":  nmse_db,
        "nmse":     nmse_metrics["nmse_linear"],
        "mse":      mse,
        "rmse":     rmse,
        "mae":      mae,
        "cos_sim":  cos_sim,
        "pearson":  pearson,
    }

    return metrics


def print_metrics(metrics, snr_db, n_samples=None):
    """Pretty-print a metrics dictionary."""
    print(f"\n{'='*55}")
    print(f"  TEST SET RESULTS — SNR = {snr_db} dB")
    if n_samples:
        print(f"  Samples tested : {n_samples}")
    print(f"{'─'*55}")
    print(f"  NMSE (dB)            : {metrics['nmse_db']:.2f} dB")
    print(f"  NMSE (linear)        : {metrics['nmse']:.6e}")
    print(f"  MSE                  : {metrics['mse']:.6e}")
    print(f"  RMSE                 : {metrics['rmse']:.6e}")
    print(f"  MAE                  : {metrics['mae']:.6e}")
    print(f"  Cosine Similarity    : {metrics['cos_sim']:.6f}")
    print(f"  Pearson Correlation  : {metrics['pearson']:.6f}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════
#  5.  TRAIN & EVALUATE FOR A SINGLE SNR
# ══════════════════════════════════════════════════════════════

def train_and_evaluate_single(snr_db, epochs, lr, batch_size, device, logger=None, run_id=None):
    """
    Full pipeline for one SNR level: build data → train → evaluate.

    Returns
    -------
    metrics : dict — evaluation results
    """
    print(f"\n{'#'*60}")
    print(f"#  SNR = {snr_db} dB")
    print(f"{'#'*60}")

    # ── Build DataLoaders ─────────────────────────────────────
    train_loader, test_loader, norm_stats = build_dataloaders(snr_db, batch_size)

    # ── Instantiate model, loss, optimizer ────────────────────
    model     = ChannelEstimatorCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = count_parameters(model)
    print(f"[INFO] Model parameters: {n_params:,}  (budget: <100k)")
    assert n_params < 100_000, f"Model exceeds 100k budget: {n_params:,}"

    # ── Training ──────────────────────────────────────────────
    print(f"\n[TRAIN] Starting training — {epochs} epochs, "
          f"lr={lr}, SNR={snr_db} dB\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = train(model, train_loader, criterion, optimizer, device, epoch, epochs)
        if logger is not None and run_id is not None:
            logger.log_step(run_id, "epoch", epoch, {"train_mse_loss": float(epoch_loss)})

    # ── Save trained model weights ────────────────────────────
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, f"cnn_{snr_db}dB.pth")
    torch.save(model.state_dict(), weight_path)
    print(f"\n[✓] Model saved → {weight_path}")

    # ── Evaluation ────────────────────────────────────────────
    metrics = evaluate(model, test_loader, device, norm_stats)
    print_metrics(metrics, snr_db, n_samples=len(test_loader.dataset))

    return metrics


# ══════════════════════════════════════════════════════════════
#  6.  COMPUTE LS BASELINE NMSE (for comparison)
# ══════════════════════════════════════════════════════════════

def compute_ls_metrics(snr_db):
    """
    Compute all evaluation metrics for the LS estimator (H_ls = H_noisy).
    Uses the raw .npy files (not the normalized CNN splits).

    Returns
    -------
    metrics : dict
    """
    dataset_dir = "../../dataset"
    H_true  = np.load(os.path.join(dataset_dir, "H_clean.npy"))
    H_noisy = np.load(os.path.join(dataset_dir, f"H_noisy_{snr_db}dB.npy"))

    # LS estimate is just the noisy observation
    H_ls = H_noisy
    N = H_true.shape[0]

    # NMSE
    nmse_num = np.linalg.norm(H_true - H_ls) ** 2
    nmse_den = np.linalg.norm(H_true) ** 2
    nmse     = nmse_num / nmse_den
    nmse_db  = 10 * np.log10(nmse)

    # MSE, RMSE, MAE
    mse  = np.mean(np.abs(H_true - H_ls) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(H_true - H_ls))

    # Cosine Similarity
    cos_sims = []
    for i in range(N):
        ht = H_true[i].flatten()
        hp = H_ls[i].flatten()
        dot    = np.real(np.vdot(ht, hp))
        norm_t = np.linalg.norm(ht)
        norm_p = np.linalg.norm(hp)
        if norm_t > 0 and norm_p > 0:
            cos_sims.append(dot / (norm_t * norm_p))
    cos_sim = float(np.mean(cos_sims))

    # Pearson Correlation (on real/imag stacked representation)
    # Stack to (N, 2, Nr, Nt) format matching CNN evaluation
    H_true_ri  = np.stack([H_true.real, H_true.imag], axis=1)
    H_ls_ri    = np.stack([H_ls.real,   H_ls.imag],   axis=1)
    pearsons = []
    for i in range(N):
        yt = H_true_ri[i].flatten()
        yp = H_ls_ri[i].flatten()
        yt_c = yt - yt.mean()
        yp_c = yp - yp.mean()
        num = np.dot(yt_c, yp_c)
        den = np.linalg.norm(yt_c) * np.linalg.norm(yp_c)
        if den > 0:
            pearsons.append(num / den)
    pearson = float(np.mean(pearsons))

    return {
        "nmse_db": nmse_db, "nmse": nmse,
        "mse": mse, "rmse": rmse, "mae": mae,
        "cos_sim": cos_sim, "pearson": pearson,
    }


# ══════════════════════════════════════════════════════════════
#  7.  SUMMARY TABLE — CNN vs LS ACROSS ALL SNRs
# ══════════════════════════════════════════════════════════════

def print_summary_table(all_cnn_metrics, all_ls_metrics, snr_levels):
    """Print a comprehensive comparison table."""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + "  COMPREHENSIVE EVALUATION : CNN vs LS  ACROSS ALL SNRs".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # ── Table 1: NMSE (dB) ────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Table 1: NMSE (dB)  — lower is better")
    print(f"{'─'*60}")
    header = f"  {'SNR (dB)':>10}  {'LS':>12}  {'CNN':>12}  {'Gain':>12}"
    print(header)
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}")
    for snr in snr_levels:
        ls_val  = all_ls_metrics[snr]["nmse_db"]
        cnn_val = all_cnn_metrics[snr]["nmse_db"]
        gain    = ls_val - cnn_val   # positive = CNN is better
        print(f"  {snr:>10d}  {ls_val:>12.2f}  {cnn_val:>12.2f}  {gain:>+11.2f} dB")

    # ── Table 2: MSE ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Table 2: MSE  — lower is better")
    print(f"{'─'*60}")
    header = f"  {'SNR (dB)':>10}  {'LS':>14}  {'CNN':>14}"
    print(header)
    print(f"  {'─'*10}  {'─'*14}  {'─'*14}")
    for snr in snr_levels:
        ls_val  = all_ls_metrics[snr]["mse"]
        cnn_val = all_cnn_metrics[snr]["mse"]
        print(f"  {snr:>10d}  {ls_val:>14.6e}  {cnn_val:>14.6e}")

    # ── Table 3: RMSE ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Table 3: RMSE  — lower is better")
    print(f"{'─'*60}")
    header = f"  {'SNR (dB)':>10}  {'LS':>14}  {'CNN':>14}"
    print(header)
    print(f"  {'─'*10}  {'─'*14}  {'─'*14}")
    for snr in snr_levels:
        ls_val  = all_ls_metrics[snr]["rmse"]
        cnn_val = all_cnn_metrics[snr]["rmse"]
        print(f"  {snr:>10d}  {ls_val:>14.6e}  {cnn_val:>14.6e}")

    # ── Table 4: MAE ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Table 4: MAE  — lower is better")
    print(f"{'─'*60}")
    header = f"  {'SNR (dB)':>10}  {'LS':>14}  {'CNN':>14}"
    print(header)
    print(f"  {'─'*10}  {'─'*14}  {'─'*14}")
    for snr in snr_levels:
        ls_val  = all_ls_metrics[snr]["mae"]
        cnn_val = all_cnn_metrics[snr]["mae"]
        print(f"  {snr:>10d}  {ls_val:>14.6e}  {cnn_val:>14.6e}")

    # ── Table 5: Cosine Similarity ────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Table 5: Cosine Similarity  — higher is better (max=1.0)")
    print(f"{'─'*60}")
    header = f"  {'SNR (dB)':>10}  {'LS':>12}  {'CNN':>12}"
    print(header)
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}")
    for snr in snr_levels:
        ls_val  = all_ls_metrics[snr]["cos_sim"]
        cnn_val = all_cnn_metrics[snr]["cos_sim"]
        print(f"  {snr:>10d}  {ls_val:>12.6f}  {cnn_val:>12.6f}")

    # ── Table 6: Pearson Correlation ──────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Table 6: Pearson Correlation  — higher is better (max=1.0)")
    print(f"{'─'*60}")
    header = f"  {'SNR (dB)':>10}  {'LS':>12}  {'CNN':>12}"
    print(header)
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}")
    for snr in snr_levels:
        ls_val  = all_ls_metrics[snr]["pearson"]
        cnn_val = all_cnn_metrics[snr]["pearson"]
        print(f"  {snr:>10d}  {ls_val:>12.6f}  {cnn_val:>12.6f}")

    print(f"\n{'═'*60}")
    print(f"  [✓] Evaluation complete for {len(snr_levels)} SNR levels.")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════
#  8.  MAIN — ORCHESTRATE FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device selection (MPS for Apple Silicon, CUDA, or CPU) ─
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    if args.all:
        # ── Train + evaluate on ALL SNR levels ────────────────
        all_cnn_metrics = {}
        all_ls_metrics  = {}

        for snr in SNR_LEVELS:
            run_id = args.run_id or f"cnn_centralized_snr{snr}_seed{args.seed}"
            logger = ExperimentLogger(args.log_dir)
            logger.start_run(
                RunMetadata(
                    run_id=run_id,
                    model="CNN",
                    setting="Centralized",
                    data_mode="N/A",
                    snr_db=snr,
                    seed=args.seed,
                    local_epochs=None,
                    global_rounds=None,
                    mu=None,
                    lr=args.lr,
                    batch_size=args.batch,
                    config_path=args.config_path or None,
                    config_sha256=file_sha256(args.config_path) if args.config_path else None,
                    git_commit=get_git_commit(str(PROJECT_ROOT)),
                    python_version=runtime_python_version(),
                    torch_version=detect_torch_version(),
                    device=str(device),
                    platform=runtime_platform(),
                )
            )
            cnn_metrics = train_and_evaluate_single(
                snr, args.epochs, args.lr, args.batch, device, logger=logger, run_id=run_id
            )
            all_cnn_metrics[snr] = cnn_metrics
            logger.end_run(run_id, {"final_nmse_db": cnn_metrics["nmse_db"], "final_nmse_linear": cnn_metrics["nmse"]})

            ls_metrics = compute_ls_metrics(snr)
            all_ls_metrics[snr] = ls_metrics

        # ── Print comprehensive summary ──────────────────────
        print_summary_table(all_cnn_metrics, all_ls_metrics, SNR_LEVELS)

    else:
        run_id = args.run_id or f"cnn_centralized_snr{args.snr}_seed{args.seed}"
        logger = ExperimentLogger(args.log_dir)
        logger.start_run(
            RunMetadata(
                run_id=run_id,
                model="CNN",
                setting="Centralized",
                data_mode="N/A",
                snr_db=args.snr,
                seed=args.seed,
                local_epochs=None,
                global_rounds=None,
                mu=None,
                lr=args.lr,
                batch_size=args.batch,
                config_path=args.config_path or None,
                config_sha256=file_sha256(args.config_path) if args.config_path else None,
                git_commit=get_git_commit(str(PROJECT_ROOT)),
                python_version=runtime_python_version(),
                torch_version=detect_torch_version(),
                device=str(device),
                platform=runtime_platform(),
            )
        )
        # ── Single SNR mode ──────────────────────────────────
        metrics = train_and_evaluate_single(
            args.snr, args.epochs, args.lr, args.batch, device, logger=logger, run_id=run_id
        )
        logger.end_run(run_id, {"final_nmse_db": metrics["nmse_db"], "final_nmse_linear": metrics["nmse"]})

        # Also show LS comparison for context
        ls_metrics = compute_ls_metrics(args.snr)
        print(f"\n  LS Baseline → NMSE = {ls_metrics['nmse_db']:.2f} dB")
        gain = ls_metrics["nmse_db"] - metrics["nmse_db"]
        print(f"  CNN  Gain   → {gain:+.2f} dB better than LS")


if __name__ == "__main__":
    main()
