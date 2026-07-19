import deepmimo as dm
import numpy as np

# 1. Load the Scenario
scen_name = 'O1_60'
dataset = dm.load(scen_name)

# 2. Select Base Station 0 (first BS) and subset to 6000 users
scenario = dataset[0]
print(f"Total users available: {scenario.n_ue}")

# Pick 6000 uniformly spaced user indices
total_users = scenario.n_ue
idxs = np.linspace(0, total_users - 1, 6000, dtype=int)
scenario = scenario.subset(idxs)
print(f"Selected users: {scenario.n_ue}")

# 3. Configure Channel Parameters
params = dm.ChannelParameters()
params.bs_antenna.shape = np.array([8, 8])   # 8x8 BS antenna array
params.ue_antenna.shape = np.array([1, 1])   # Single UE antenna
params.freq_domain = 1
params.ofdm.bandwidth = 0.5e6               # 0.5 MHz bandwidth
params.ofdm.subcarriers = 64
params.ofdm.selected_subcarriers = np.arange(64)

# 4. Generate the Channel Matrix
print("Generating channels for 6000 users...")
scenario.compute_channels(params)

# 5. Export Dataset
h_dataset = scenario.channels
np.save('o1_60_matrix.npy', h_dataset)
print(f"Success! Final Shape: {h_dataset.shape}")
print(f"Saved to o1_60_matrix.npy")

#inspecting the Data Set 
data = np.load("o1_60_matrix.npy")

print("Dataset shape:", data.shape)

users = data.shape[0]
rx_ant = data.shape[1]
tx_ant = data.shape[2]
subcarriers = data.shape[3]

print("Users:", users)
print("RX antennas:", rx_ant)
print("TX antennas:", tx_ant)
print("Subcarriers:", subcarriers)