import numpy as np
import matplotlib.pyplot as plt
import torch

def set_torch_precision(use_float64=False):
    """
    Set the default PyTorch tensor precision globally.
    
    Args:
        use_float64 (bool): If True, use float64 (double precision), 
                           if False, use float32 (single precision)
    """
    if use_float64:
        torch.set_default_dtype(torch.float64)
        print("Set PyTorch default precision to float64 (double precision)")
    else:
        torch.set_default_dtype(torch.float32)
        print("Set PyTorch default precision to float32 (single precision)")
    
    return torch.get_default_dtype()

def plot_example_trajectories( x_trunk, y, num_samples=4, x_branch=None): 
    # Plot example trajectories
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        if x_branch is not None:
            plt.plot(x_trunk[i], x_branch[i], label=f"Ext force {i+1}", linestyle='dashed')
        for j in range(y.shape[1]):
            plt.plot(x_trunk[i], y[i,j], label=f"Sample {i+1} - {j+1}", linestyle='solid')

    plt.xlabel("Time")
    plt.ylabel("Displacement x(t)")
    plt.legend()
    plt.title("Sample trajectories of the forced oscillator")
    plt.show()
    return

def set_time(end_time, interval_points):
    t_span = (0, end_time)
    t_eval = np.linspace(0, end_time, interval_points)
    return t_span, t_eval

def plot_traj(time_all, solution, ext_series,pointer=0):
    sol = solution[pointer]
    time = time_all[pointer]
    plt.figure(figsize=(10, 5))
    plt.plot(time, ext_series[pointer][0], label='Vs')
    plt.plot(time, sol[0], label='delta')
    plt.plot(time, sol[1], label='omega')
    plt.plot(time, sol[2], label='Eq')
    plt.plot(time, sol[3], label='Ed')
    plt.xlabel('Time (s)')
    plt.ylabel('State variables')
    plt.title('State variables vs Time')
    plt.legend()
    plt.grid()
    plt.show()
