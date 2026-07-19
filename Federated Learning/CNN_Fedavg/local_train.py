"""
local_train.py — Local client training for Federated Learning

Strategy:
  Use Adam with a modest learning rate. Although Adam's adaptive state
  resets each round, with a small LR (0.001) and just a few local epochs,
  the initial rapid adaptation phase of Adam actually helps each client
  make efficient use of its limited data per round.

  We also apply gradient clipping for stability.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cnn_model import ChannelDataset


def train_local_model(model, client_id, x_path, y_path,
                      epochs=3, batch_size=32, lr=0.001, device='cpu'):
    """
    Train a model locally on one client's data for a few epochs.

    Returns
    -------
    local_weights : dict — updated model state_dict (cpu tensors)
    n_samples     : int  — number of training samples this client has
    """
    print(f"--- Client {client_id} Local Training ---")

    # Load client data
    try:
        dataset = ChannelDataset(x_path, y_path)
    except Exception as e:
        print(f"Error loading data for client {client_id}: {e}")
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_samples = len(dataset)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)

            loss.backward()

            # Gradient clipping — stabilises training after weight aggregation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Client {client_id} | Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.6f}")

    # Return updated weights on CPU
    local_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return local_weights, n_samples
