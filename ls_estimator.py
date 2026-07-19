import numpy as np
import os
import matplotlib.pyplot as plt

dataset_dir = "dataset"

snr_levels = [0,5,10,15,20]

# Load clean channel
H_true = np.load(os.path.join(dataset_dir,"H_clean.npy"))

print("Clean channel shape:",H_true.shape)

def compute_nmse(H_true,H_est):

    num = np.linalg.norm(H_true - H_est)**2
    den = np.linalg.norm(H_true)**2

    return num/den


results = []

for snr in snr_levels:

    H_noisy = np.load(os.path.join(dataset_dir,f"H_noisy_{snr}dB.npy"))

    # LS estimator
    H_ls = H_noisy

    nmse = compute_nmse(H_true,H_ls)

    nmse_db = 10*np.log10(nmse)

    results.append(nmse_db)

    print(f"SNR {snr} dB → NMSE: {nmse_db:.2f} dB")


print("\nLS estimation finished.")