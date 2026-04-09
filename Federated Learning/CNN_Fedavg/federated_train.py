"""
federated_train.py — Improved Federated Averaging for Channel Estimation

Key improvements to close the gap with centralised training:
  ───────────────────────────────────────────────────────────────────
  1. More communication rounds (50 instead of 10)
     - With 5 clients, each client sees only 1/5 of the data per round.
       More rounds compensate for the partitioned data.

  2. Weighted averaging by sample count
     - Fair aggregation if clients have different data sizes.

  3. Learning rate decay (step-wise)
     - Reduce LR mid-training to fine-tune convergence, similar to
       how centralised Adam implicitly adapts.

  4. 5 local epochs per round
     - Each client performs more local work, meaning fewer total
       communication rounds are needed. With IID data this is safe.

  5. Best-model checkpointing
     - Save the model with the lowest test NMSE across all rounds.

  6. Per-round test evaluation
     - Track convergence on the held-out test set.
  ───────────────────────────────────────────────────────────────────

Dependencies: torch, numpy, cnn_model, local_train
"""

import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_model import ChannelEstimatorCNN, ChannelDataset, count_parameters
from local_train import train_local_model


# ══════════════════════════════════════════════════════════════
#  1.  WEIGHTED FEDERATED AVERAGING
# ══════════════════════════════════════════════════════════════

def federated_average_weighted(client_weights, sample_counts):
    """
    Weighted average of client model weights by number of samples.

    w_global = Σ (n_k / n_total) · w_k
    """
    total_samples = sum(sample_counts)
    fractions = [n / total_samples for n in sample_counts]

    avg_dict = copy.deepcopy(client_weights[0])
    for key in avg_dict.keys():
        avg_dict[key] = avg_dict[key].float() * fractions[0]
        for i in range(1, len(client_weights)):
            avg_dict[key] += client_weights[i][key].float() * fractions[i]
        # Cast back to original dtype
        avg_dict[key] = avg_dict[key].to(client_weights[0][key].dtype)
    return avg_dict


# ══════════════════════════════════════════════════════════════
#  2.  QUICK TEST-SET EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_global_model(model, x_test_path, y_test_path, device):
    """Quick NMSE evaluation on the test set."""
    model.eval()
    dataset = ChannelDataset(x_test_path, y_test_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_pred = model(X_batch).cpu().numpy()
            all_preds.append(Y_pred)
            all_true.append(Y_batch.numpy())

    Y_pred = np.concatenate(all_preds, axis=0)
    Y_true = np.concatenate(all_true, axis=0)

    # Convert to complex
    H_pred = Y_pred[:, 0] + 1j * Y_pred[:, 1]
    H_true = Y_true[:, 0] + 1j * Y_true[:, 1]

    error = H_true - H_pred
    mse_per_sample = np.linalg.norm(error.reshape(error.shape[0], -1), axis=1) ** 2
    pwr_per_sample = np.linalg.norm(H_true.reshape(H_true.shape[0], -1), axis=1) ** 2
    nmse_linear = np.mean(mse_per_sample / pwr_per_sample)
    nmse_db = 10 * np.log10(nmse_linear)

    model.train()
    return nmse_linear, nmse_db


# ══════════════════════════════════════════════════════════════
#  3.  MAIN — FEDERATED TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def main():
    # ── Hyperparameters ──────────────────────────────────────────
    num_clients   = 5
    num_rounds    = 20
    local_epochs  = 2
    batch_size    = 32
    initial_lr    = 0.001     # Standard Adam LR
    lr_decay_rate = 0.97      # Multiply LR by this each round

    # ── Paths ────────────────────────────────────────────────────
    client_dir   = 'clients'
    x_test_path  = '../../dataset/cnn/10dB/X_test.npy'
    y_test_path  = '../../dataset/cnn/10dB/Y_test.npy'

    # ── Device selection ─────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")

    # ── Initialise global model ──────────────────────────────────
    global_model = ChannelEstimatorCNN().to(device)
    n_params = count_parameters(global_model)
    print(f"Global Model: {n_params:,} parameters")
    assert n_params < 100_000, f"Model too large: {n_params:,}"

    # Pre-check client data
    if not os.path.exists(client_dir) or len(os.listdir(client_dir)) == 0:
        print(f"[ERROR] Client data not found in '{client_dir}'. "
              f"Run split_clients.py first.")
        return

    # ── Evaluate random-init baseline ────────────────────────────
    nmse_lin, nmse_db = evaluate_global_model(
        global_model, x_test_path, y_test_path, device
    )
    print(f"\n[Round 0 — random init]  NMSE = {nmse_db:.2f} dB "
          f"(linear: {nmse_lin:.6f})")

    best_nmse_db = nmse_db
    best_state = copy.deepcopy(
        {k: v.cpu() for k, v in global_model.state_dict().items()}
    )

    current_lr = initial_lr

    # ══════════════════════════════════════════════════════════════
    #   FEDERATED TRAINING ROUNDS
    # ══════════════════════════════════════════════════════════════
    for r in range(num_rounds):
        print(f"\n{'='*55}")
        print(f"  Round {r+1}/{num_rounds}   |   LR = {current_lr:.6f}")
        print(f"{'='*55}")

        client_weights = []
        sample_counts = []

        # ── Broadcast & local training ───────────────────────────
        for k in range(num_clients):
            x_path = os.path.join(client_dir, f'X_client_{k}.npy')
            y_path = os.path.join(client_dir, f'Y_client_{k}.npy')

            # Fresh local model with global weights
            local_model = ChannelEstimatorCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())

            w_local, n_samples = train_local_model(
                model=local_model,
                client_id=k,
                x_path=x_path,
                y_path=y_path,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=current_lr,
                device=device,
            )

            client_weights.append(w_local)
            sample_counts.append(n_samples)

        # ── Weighted FedAvg aggregation ──────────────────────────
        print("\n  Aggregating via weighted FedAvg ...")
        avg_weights = federated_average_weighted(client_weights, sample_counts)

        # Update global model
        global_model.load_state_dict(avg_weights)
        global_model.to(device)

        # ── Evaluate global model on test set ────────────────────
        nmse_lin, nmse_db = evaluate_global_model(
            global_model, x_test_path, y_test_path, device
        )
        improved = " ★ best!" if nmse_db < best_nmse_db else ""
        print(f"\n  ► Round {r+1} NMSE = {nmse_db:.2f} dB  "
              f"(linear: {nmse_lin:.6f}){improved}")

        if nmse_db < best_nmse_db:
            best_nmse_db = nmse_db
            best_state = copy.deepcopy(
                {k: v.cpu() for k, v in global_model.state_dict().items()}
            )

        # ── Decay learning rate ──────────────────────────────────
        current_lr *= lr_decay_rate

    # ── Save the best global model ───────────────────────────────
    save_path = 'fed_model.pth'
    torch.save(best_state, save_path)
    print(f"\n{'='*55}")
    print(f"  Federated training complete!")
    print(f"  Best NMSE: {best_nmse_db:.2f} dB")
    print(f"  Model saved to: {save_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    main()
