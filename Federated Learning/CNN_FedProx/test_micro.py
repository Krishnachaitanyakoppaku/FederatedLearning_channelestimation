import os
import torch
import numpy as np
import sys
sys.path.append(".")
from federated_train import run_experiment

def test():
    torch.manual_seed(42)
    np.random.seed(42)
    fedavg_db, _ = run_experiment("FedAvg E=2", "clients_non_iid", use_fedprox=False, local_epochs=2, num_rounds=10)
    
    for mu in [1e-3, 1e-4, 5e-5, 1e-5, 1e-6]:
        torch.manual_seed(42)
        np.random.seed(42)
        fp_db, _ = run_experiment(f"FedProx mu={mu}", "clients_non_iid", use_fedprox=True, local_epochs=2, mu=mu, num_rounds=10)
        print(f"mu={mu} -> {fp_db:.4f} dB (FedAvg: {fedavg_db:.4f} dB)")
        if fp_db < fedavg_db:
            print("FOUND BETTER MU!")

if __name__ == "__main__":
    test()
