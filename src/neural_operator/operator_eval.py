import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset



import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime

def save_plot_all_derivatives(
    model, dataset, modelling_full, calculate_autograd, calculate_from_ode,
    device='cpu', num_samples=-1, time_slice=None, epoch=None,
    output_dir="training_plots", run_number=None, save_flag=True
):
    """
    Plot and save true vs predicted derivatives and states for multiple samples.

    Parameters:
        model: Trained model.
        dataset: PyTorch dataset or DataLoader.
        modelling_full: ODE function to compute RHS.
        calculate_autograd: Function to compute dy/dt from model with autograd.
        calculate_from_ode: Function to compute RHS from output or target.
        device (str): Device to use ('cpu' or 'cuda').
        num_samples (int): Number of samples to use; -1 for all.
        time_slice (tuple or None): Slice time series as (start, end).
        epoch (int or None): Appends to filenames if given.
        output_dir (str): Base directory to save plots.
        run_number (int or None): Training run number; created automatically if None.
    """
    model.eval()
    if isinstance(dataset, DataLoader):
        data_loader = dataset
    else:
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    branch_input_all, trunk_input_all, target_all = next(iter(data_loader))

    total_samples = branch_input_all.shape[0]
    if num_samples == -1 or num_samples > total_samples:
        num_samples = total_samples

    branch_input_all = branch_input_all[:num_samples].to(device)
    trunk_input_all = trunk_input_all[:num_samples].to(device)
    target_all = target_all[:num_samples].to(device)

    with torch.no_grad():
        predictions, dy_dt_all = calculate_autograd(model, branch_input_all, trunk_input_all)
        rhs_all = calculate_from_ode(modelling_full, predictions, branch_input_all)
        rhs_true_all = calculate_from_ode(modelling_full, target_all, branch_input_all)

    print(f"Total loss: {torch.mean((target_all - predictions) ** 2).item():.4e},Physics loss: {torch.mean((dy_dt_all - rhs_true_all) ** 2).item():.4e}", )
    # calculate mae also
    print(f"MAE: {torch.mean(torch.abs(target_all - predictions)).item():.4e}, Physics MAE: {torch.mean(torch.abs(dy_dt_all - rhs_true_all)).item():.4e}", )
    # Move to CPU
    predictions = predictions.cpu().detach()
    dy_dt_all = dy_dt_all.cpu().detach()
    rhs_all = rhs_all.cpu().detach()
    rhs_true_all = rhs_true_all.cpu().detach()
    targets = target_all.cpu().detach()
    time = trunk_input_all[0].squeeze(-1).cpu().detach()

    if time_slice:
        start, end = time_slice
        time = time[start:end]
    else:
        start, end = 0, time.shape[0]

    # --- Folder setup ---
    if run_number is None:
        run_number = 1
        while True:
            save_path = os.path.join(output_dir, f"folder_{run_number}")
            if not os.path.exists(save_path):
                break
            run_number += 1
    else:
        save_path = os.path.join(output_dir, f"folder_{run_number}")
    os.makedirs(save_path, exist_ok=True)

    # --- Plot 1: dy/dt vs rhs_true ---
    fig1 = plt.figure(figsize=(16, 10))
    for i in range(predictions.shape[-1]):
        plt.subplot(2, 2, i + 1)
        for j in range(num_samples):
            plt.plot(time, rhs_true_all[j, start:end, i], linestyle='--', alpha=0.5, label='True' if j == 0 else "")
            plt.plot(time, dy_dt_all[j, start:end, i], alpha=0.5, label='dy/dt' if j == 0 else "")
        plt.title(f'd(State {i + 1})/dt')
        plt.xlabel('Time')
        plt.ylabel('Derivative')
        plt.legend()
    plt.tight_layout()
    if save_flag:   
        fig1.savefig(os.path.join(save_path, f"{epoch}_derivatives_comparison.png"))
        plt.close(fig1)
    else:
        plt.show()
    

    # --- Plot 2: Derivative Error (dy_dt - rhs_true) ---
    fig2 = plt.figure(figsize=(16, 10))
    for i in range(predictions.shape[-1]):
        plt.subplot(2, 2, i + 1)
        for j in range(num_samples):
            error = rhs_true_all[j, start:end, i] - dy_dt_all[j, start:end, i]
            plt.plot(time, error, linestyle='--', alpha=0.5, label='Error' if j == 0 else "")
        plt.title(f'Error in d(State {i + 1})/dt')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.legend()
    plt.tight_layout()
    if save_flag:   
        fig2.savefig(os.path.join(save_path, f"{epoch}_derivatives_error.png"))
        plt.close(fig2)
    else:
        plt.show()
    # --- Plot 3: Predicted vs True State Variables ---
    fig3 = plt.figure(figsize=(16, 10))
    for i in range(predictions.shape[-1]):
        plt.subplot(2, 2, i + 1)
        for j in range(num_samples):
            plt.plot(time, targets[j, start:end, i], linestyle='--', alpha=0.5, label='True' if j == 0 else "")
            plt.plot(time, predictions[j, start:end, i], alpha=0.5, label='Predicted' if j == 0 else "")
        plt.title(f'State Variable {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    if save_flag:   
        fig3.savefig(os.path.join(save_path, f"{epoch}_states_comparison.png"))
        plt.close(fig3)
    else:
        plt.show()

    # --- Plot 4: State Variable Error ---
    fig4 = plt.figure(figsize=(16, 10))
    for i in range(predictions.shape[-1]):
        plt.subplot(2, 2, i + 1)
        for j in range(num_samples):
            plt.plot(time, targets[j, start:end, i] - predictions[j, start:end, i],
                     linestyle='--', alpha=0.5, label='Error' if j == 0 else "")
        plt.title(f'Error in State Variable {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.legend()
    plt.tight_layout()
    if save_flag:   
        fig4.savefig(os.path.join(save_path, f"{epoch}_states_error.png"))
        plt.close(fig4)
    else:
        plt.show()
   
    
    return run_number
