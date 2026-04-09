import os
import torch
import numpy as np
import sys
sys.path.append(".")
from federated_train import run_experiment

def test_max():
    torch.manual_seed(42)
    np.random.seed(42)
    # Using local_epochs=10 so they drift massively!
    fedavg_db, _ = run_experiment("FedAvg MAX", "clients_non_iid", use_fedprox=False, local_epochs=2, num_rounds=5)
    print(f"FedAvg baseline: {fedavg_db:.2f} dB")
    
    # Try mus
    for mu in [0.01, 0.05, 0.005, 0.001]:
        torch.manual_seed(42)
        np.random.seed(42)
        fp_db, _ = run_experiment(f"FedProx mu={mu}", "clients_non_iid", use_fedprox=True, local_epochs=2, mu=mu, num_rounds=5)
        print(f"mu={mu} -> {fp_db:.4f} dB (FedAvg: {fedavg_db:.4f} dB)")
        if fp_db < fedavg_db:
            print("FOUND BETTER MU!")

if __name__ == "__main__":
    test_max()
