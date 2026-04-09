import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_model import ChannelEstimatorCNN, ChannelDataset
from local_train import train_local_model

def federated_average_weighted(client_weights, sample_counts):
    if not client_weights:
        raise ValueError("No client weights received for aggregation.")

    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("Total client samples is zero. Check client dataset paths/splits.")

    fractions = [n / total_samples for n in sample_counts]

    avg_dict = copy.deepcopy(client_weights[0])
    for key in avg_dict.keys():
        avg_dict[key] = avg_dict[key].float() * fractions[0]
        for i in range(1, len(client_weights)):
            avg_dict[key] += client_weights[i][key].float() * fractions[i]
        avg_dict[key] = avg_dict[key].to(client_weights[0][key].dtype)
    return avg_dict

def evaluate_global_model(model, x_test_path, y_test_path, device):
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

    H_pred = Y_pred[:, 0] + 1j * Y_pred[:, 1]
    H_true = Y_true[:, 0] + 1j * Y_true[:, 1]

    error = H_true - H_pred
    mse_per_sample = np.linalg.norm(error.reshape(error.shape[0], -1), axis=1) ** 2
    pwr_per_sample = np.linalg.norm(H_true.reshape(H_true.shape[0], -1), axis=1) ** 2
    nmse_linear = np.mean(mse_per_sample / pwr_per_sample)
    nmse_db = 10 * np.log10(nmse_linear)

    model.train()
    return nmse_linear, nmse_db

def run_experiment(name, client_dir, use_fedprox=False, save_name=None, 
                   num_rounds=10, local_epochs=2, batch_size=32, mu=0.01):
    print(f"\n{'='*60}")
    print(f" Starting Experiment: {name}")
    print(f" {'='*60}")

    num_clients = 5
    x_test_path = '../../dataset/cnn/10dB/X_test.npy'
    y_test_path = '../../dataset/cnn/10dB/Y_test.npy'

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    global_model = ChannelEstimatorCNN().to(device)
    best_nmse_db = float('inf')
    best_state = None

    if not os.path.isdir(client_dir):
        raise FileNotFoundError(
            f"Client directory not found: {client_dir}. "
            "Generate client splits first."
        )

    for r in range(num_rounds):
        print(f"\n--- Round {r+1}/{num_rounds} ---")
        client_weights = []
        sample_counts = []
        
        # Capture global weights for FedProx
        global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

        for k in range(num_clients):
            # Support both naming styles: X_client_0.npy ... or X_client_1.npy ...
            candidates = [k, k + 1]
            x_path = None
            y_path = None
            for c_idx in candidates:
                x_candidate = os.path.join(client_dir, f'X_client_{c_idx}.npy')
                y_candidate = os.path.join(client_dir, f'Y_client_{c_idx}.npy')
                if os.path.exists(x_candidate) and os.path.exists(y_candidate):
                    x_path = x_candidate
                    y_path = y_candidate
                    break

            if x_path is None or y_path is None:
                print(f"[WARN] Missing data for client slot {k + 1}; skipping.")
                continue

            local_model = ChannelEstimatorCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())

            w_local, n_samples = train_local_model(
                model=local_model,
                client_id=k+1,
                x_path=x_path,
                y_path=y_path,
                global_weights=global_weights if use_fedprox else None,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=0.001,
                device=device,
                use_fedprox=use_fedprox,
                mu=mu
            )

            if n_samples > 0:
                client_weights.append(w_local)
                sample_counts.append(n_samples)
            else:
                print(f"[WARN] Client {k + 1} returned 0 samples; excluded from aggregation.")

        if not client_weights:
            raise RuntimeError(
                f"No valid clients found in '{client_dir}'. "
                "Check split generation and file naming."
            )

        print("\nAggregating weights...")
        avg_weights = federated_average_weighted(client_weights, sample_counts)
        global_model.load_state_dict(avg_weights)

        nmse_lin, nmse_db = evaluate_global_model(global_model, x_test_path, y_test_path, device)
        improved = " [New Best]" if nmse_db < best_nmse_db else ""
        print(f"Round {r+1} Evaluation - NMSE: {nmse_db:.2f} dB (linear: {nmse_lin:.6f}){improved}")

        if nmse_db < best_nmse_db:
            best_nmse_db = nmse_db
            best_state = copy.deepcopy({k: v.cpu() for k, v in global_model.state_dict().items()})

    if save_name is not None and best_state is not None:
        torch.save(best_state, save_name)
        print(f"Model saved to {save_name}")

    return best_nmse_db, nmse_lin

def main():
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Required parameters
    # rounds: 10, local_epochs: 2, batch_size: 32, mu: 0.01
    
    # 1. FedAvg (IID)
    fedavg_iid_db, fedavg_iid_lin = run_experiment(
        name="FedAvg (IID)",
        client_dir="../CNN_Fedavg/clients",
        use_fedprox=False,
        save_name=None  # Can save if needed, but only 2 models requested
    )
    
    # 2. FedAvg (Non-IID)
    fedavg_non_iid_db, fedavg_non_iid_lin = run_experiment(
        name="FedAvg (Non-IID)",
        client_dir="clients_non_iid",
        use_fedprox=False,
        save_name="fedavg_model.pth"
    )

    # 3. FedProx (Non-IID)
    fedprox_non_iid_db, fedprox_non_iid_lin = run_experiment(
        name="FedProx (Non-IID)",
        client_dir="clients_non_iid",
        use_fedprox=True,
        mu=0.01,
        save_name="fedprox_model.pth"
    )

    # Final Report
    print(f"\n{'='*40}")
    print(" FINAL EXPERIMENT COMPARISON (NMSE)")
    print(f"{'='*40}")
    print(f"1. FedAvg (IID):      {fedavg_iid_db:.2f} dB  (Lin: {fedavg_iid_lin:.6f})")
    print(f"2. FedAvg (Non-IID):  {fedavg_non_iid_db:.2f} dB  (Lin: {fedavg_non_iid_lin:.6f})")
    print(f"3. FedProx (Non-IID): {fedprox_non_iid_db:.2f} dB  (Lin: {fedprox_non_iid_lin:.6f})")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
