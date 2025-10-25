# Train DeepONet function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pytorch_optimizer import SOAP
import os
import torch
from torch.utils.data import Dataset
from src.neural_operator.operator_eval import save_plot_all_derivatives

class CollocationDeepONetDataset(Dataset):
    def __init__(self, data, config):
        self.samples = []
        self.ic_samples = []
        # Set precision based on flag
        self.config = config
        use_float64 = config.flag_for_precision
        self.dtype = torch.float64 if use_float64 else torch.float32

        for sample in data:
            time = torch.tensor(sample[0], dtype=self.dtype).unsqueeze(-1)       # [1000, 1]
            x0 = torch.tensor(sample[1], dtype=self.dtype)    
            force1 = torch.tensor(sample[2], dtype=self.dtype)  
            force2 = torch.tensor(sample[3], dtype=self.dtype)                   # [1000]

            # Concatenate all forces into one flat vector: [2 × 1000]
            force = torch.cat([force1, force2], dim=0)                              # [2000]

            # Branch input: [4] + [2000] = [2004]
            branch_input = torch.cat([x0, force], dim=0)                            # [2004]

            # Output: stack responses into [1000, 4]

            self.samples.append((branch_input, time))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        branch_input, trunk_input = self.samples[idx]
        return branch_input, trunk_input

    def create_path(self):
        import os
        save_path = self.config.dirs.dataset_dir + "/" + self.config.model.model_flag+ "/" 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def save_hd5f(self, filename):
        """
        Save the dataset to HDF5 format.
        
        Args:
            filename (str): Path to save the HDF5 file
        """
        import h5py
        save_path = self.create_path()
        existing_files = [f for f in os.listdir(save_path)  if f.endswith('.h5') and (filename+'_initial_conditions' in f)]   
        file_count = len(existing_files)
        save_path_init = save_path + f"{filename}_initial_conditions_dataset_{file_count+1}.h5"

        save_path = save_path + f"{filename}_dataset_{file_count+1}.h5"

        with h5py.File(save_path, 'w') as f:
            # Create groups for better organization
            f.attrs['num_samples'] = len(self.samples)
            f.attrs['dtype'] = str(self.dtype)
            
            # Save each sample
            for i, (branch_input, trunk_input) in enumerate(self.samples):
                sample_group = f.create_group(f'sample_{i}')
                
                # Save branch input [2004]
                sample_group.create_dataset('branch_input', data=branch_input.numpy())
                
                # Save trunk input (time) [1000, 1]
                sample_group.create_dataset('trunk_input', data=trunk_input.numpy())
                
                
                # Add metadata for this sample
                sample_group.attrs['branch_input_shape'] = branch_input.shape
                sample_group.attrs['trunk_input_shape'] = trunk_input.shape
        
        
        with h5py.File(save_path_init, 'w') as f:
            # Create groups for better organization
            f.attrs['num_samples'] = len(self.samples)
            f.attrs['dtype'] = str(self.dtype)
            
            # Save each sample
            for i, (branch_input, trunk_input) in enumerate(self.samples):
                sample_group = f.create_group(f'sample_{i}')
                
                # Save branch input [2004]
                sample_group.create_dataset('branch_input', data=branch_input.numpy())
                
                # Save trunk input (time) [1000, 1]
                sample_group.create_dataset('trunk_input', data=trunk_input[0].unsqueeze(1).numpy())
                
                sample_group.create_dataset('output', data=branch_input[:4].unsqueeze(0).numpy())
                # Add metadata for this sample
                sample_group.attrs['branch_input_shape'] = branch_input.shape
                sample_group.attrs['trunk_input_shape'] = trunk_input.shape
                sample_group.attrs['output_shape'] = (4,)
        print(f"Dataset saved to {filename+'initial_conditions'} with {len(self.samples)} samples")

class TrajectoryDeepONetDataset(Dataset):
    def __init__(self, data, config):
        self.samples = []
        # Set precision based on flag
        self.config = config
        use_float64 = config.flag_for_precision
        self.dtype = torch.float64 if use_float64 else torch.float32

        for sample in data:
            time = torch.tensor(sample[0], dtype=self.dtype).unsqueeze(-1)       # [1000, 1]
            x0 = torch.tensor(sample[1], dtype=self.dtype)    
            force1 = torch.tensor(sample[2], dtype=self.dtype)  
            force2 = torch.tensor(sample[3], dtype=self.dtype)                   # [1000]
            response_list = [torch.tensor(r, dtype=self.dtype) for r in sample[4:]]  # 4 × [1000]

            # Concatenate all forces into one flat vector: [2 × 1000]
            force = torch.cat([force1, force2], dim=0)                              # [2000]

            # Branch input: [4] + [2000] = [2004]
            branch_input = torch.cat([x0, force], dim=0)                            # [2004]

            # Output: stack responses into [1000, 4]
            response = torch.stack(response_list, dim=1)                            # [1000, 4]

            self.samples.append((branch_input, time, response))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        branch_input, trunk_input, output = self.samples[idx]
        return branch_input, trunk_input, output

    def create_path(self):
        import os
        save_path = self.config.dirs.dataset_dir + "/" + self.config.model.model_flag+ "/" 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def save_hd5f(self, filename):
        """
        Save the dataset to HDF5 format.
        
        Args:
            filename (str): Path to save the HDF5 file
        """
        import h5py
        save_path = self.create_path()
        existing_files = [f for f in os.listdir(save_path)  if f.endswith('.h5') and filename in f]   
        file_count = len(existing_files)
        save_path = save_path + f"{filename}_dataset_{file_count+1}.h5"

        with h5py.File(save_path, 'w') as f:
            # Create groups for better organization
            f.attrs['num_samples'] = len(self.samples)
            f.attrs['dtype'] = str(self.dtype)
            
            # Save each sample
            for i, (branch_input, trunk_input, output) in enumerate(self.samples):
                sample_group = f.create_group(f'sample_{i}')
                
                # Save branch input [2004]
                sample_group.create_dataset('branch_input', data=branch_input.numpy())
                
                # Save trunk input (time) [1000, 1]
                sample_group.create_dataset('trunk_input', data=trunk_input.numpy())
                
                # Save output (response) [1000, 4]
                sample_group.create_dataset('output', data=output.numpy())
                
                # Add metadata for this sample
                sample_group.attrs['branch_input_shape'] = branch_input.shape
                sample_group.attrs['trunk_input_shape'] = trunk_input.shape
                sample_group.attrs['output_shape'] = output.shape
        
        print(f"Dataset saved to {filename} with {len(self.samples)} samples")


def load_hd5f(filename, config):
    """
    Load dataset from HDF5 format.
    
    Args:
        filename (str): Path to the HDF5 file
        config: Configuration object with flag_for_precision attribute
        
    Returns:
        TrajectoryDeepONetDataset: Loaded dataset
    """
    import h5py
    
    # Create empty dataset
    if "col" in filename:
        dataset = CollocationDeepONetDataset([], config)
    else:
        dataset = TrajectoryDeepONetDataset([], config)
    path = dataset.create_path()
    filename = path + filename
    with h5py.File(filename, 'r') as f:
        num_samples = f.attrs['num_samples']
        
        for i in range(num_samples):
            sample_group = f[f'sample_{i}']
            
            # Load data and convert to tensors with correct dtype
            branch_input = torch.tensor(sample_group['branch_input'][:], dtype=dataset.dtype)
            trunk_input = torch.tensor(sample_group['trunk_input'][:], dtype=dataset.dtype)
            if "col" in filename:
                dataset.samples.append((branch_input, trunk_input))
            else:
                output = torch.tensor(sample_group['output'][:], dtype=dataset.dtype)
                dataset.samples.append((branch_input, trunk_input, output))

    print(f"Dataset loaded from {filename} with {len(dataset.samples)} samples")
    return dataset

def load_hd5f_efficient(filename, config):
    """
    Load dataset from HDF5 format and return as tensors.
    
    Args:
        filename (str): Path to the HDF5 file
        config: Configuration object with flag_for_precision attribute
        
    Returns:
        tuple: (branch_inputs, trunk_inputs) for collocation data or
               (branch_inputs, trunk_inputs, outputs) for trajectory data
               Each tensor has shape [num_samples, ...] where ... is the original shape
    """
    import h5py
    
    # Create empty dataset to get path and dtype
    if "col" in filename and "initial_conditions" not in filename:
        dataset = CollocationDeepONetDataset([], config)

    else:
        dataset = TrajectoryDeepONetDataset([], config)
    path = dataset.create_path()
    filename = path + filename
    
    # Lists to collect all data
    all_branch_inputs = []
    all_trunk_inputs = []
    all_outputs = []
    
    with h5py.File(filename, 'r') as f:
        num_samples = f.attrs['num_samples']
        
        for i in range(num_samples):
            sample_group = f[f'sample_{i}']
            
            # Load data and convert to tensors with correct dtype
            branch_input = torch.tensor(sample_group['branch_input'][:], dtype=dataset.dtype)
            trunk_input = torch.tensor(sample_group['trunk_input'][:], dtype=dataset.dtype)
            
            all_branch_inputs.append(branch_input)
            all_trunk_inputs.append(trunk_input)
            
            if "col" not in filename or "initial_conditions" in filename:
                output = torch.tensor(sample_group['output'][:], dtype=dataset.dtype)
                all_outputs.append(output)

    # Stack all data into single tensors
    branch_inputs_tensor = torch.stack(all_branch_inputs)
    trunk_inputs_tensor = torch.stack(all_trunk_inputs)
    
    if "col" in filename and "initial_conditions" not in filename:
        return branch_inputs_tensor, trunk_inputs_tensor
    else:
        outputs_tensor = torch.stack(all_outputs)
        return branch_inputs_tensor, trunk_inputs_tensor, outputs_tensor

class DeepONetGPUDataset(Dataset):
    def __init__(self, Xb, Xt, Y=None):
        self.Xb, self.Xt, self.Y = Xb, Xt, Y
        self.has_y = Y is not None
    def __len__(self):
        return self.Xb.shape[0]
    def __getitem__(self, idx):
        if self.has_y:
            return self.Xb[idx], self.Xt[idx], self.Y[idx]   # GPU views, no copy
        return self.Xb[idx], self.Xt[idx]


