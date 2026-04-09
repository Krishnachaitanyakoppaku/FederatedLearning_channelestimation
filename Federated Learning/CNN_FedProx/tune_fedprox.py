import os
import torch
import numpy as np
import sys
sys.path.append(".")
from federated_train import run_experiment

def tune():
    # 1. FedAvg Non-IID
    torch.manual_seed(42)
    np.random.seed(42)
    fedavg_db, _ = run_experiment("FedAvg (Non-IID)", "clients_non_iid", use_fedprox=False, num_rounds=10)
    print(f"FedAvg baseline: {fedavg_db:.2f} dB")
    
    # 2. Try mus
    for mu in [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]:
        torch.manual_seed(42)
        np.random.seed(42)
        fp_db, _ = run_experiment(f"FedProx mu={mu}", "clients_non_iid", use_fedprox=True, mu=mu, num_rounds=10)
        print(f"mu={mu} -> {fp_db:.2f} dB (diff: {fp_db - fedavg_db:.2f})")
        if fp_db < fedavg_db:
            print("FOUND BETTER MU!")

if __name__ == "__main__":
    tune()
