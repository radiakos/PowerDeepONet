import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(input)


    





class DeepONetStackedf(nn.Module):
    def __init__(self, ic_dim, force_dim, trunk_input_dim, output_dim, hidden_dim=64, num_hidden_layers=3, act_func='tanh'):
        super(DeepONetStackedf, self).__init__()
        self.ic_dim = ic_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Select activation function class, not an instance
        if act_func == 'tanh':
            activation = nn.Tanh
        elif act_func == 'relu':
            activation = nn.ReLU
        elif act_func == 'sin':
            activation = Sine
        else:
            raise ValueError(f"Unsupported activation: {act_func}")

        # Build IC branch network with flexible number of hidden layers
        # Output dimension should be hidden_dim for proper DeepONet
        ic_layers = []
        ic_layers.append(nn.Linear(ic_dim, hidden_dim))
        ic_layers.append(activation())
        
        for _ in range(num_hidden_layers - 1):
            ic_layers.append(nn.Linear(hidden_dim, hidden_dim))
            ic_layers.append(activation())
        
        ic_layers.append(nn.Linear(hidden_dim, hidden_dim))  # Output for dot product
        self.ic_branch_net = nn.Sequential(*ic_layers)
        
        # Build force branch network with flexible number of hidden layers
        # Output dimension should be hidden_dim for proper DeepONet
        force_layers = []
        force_layers.append(nn.Linear(force_dim, hidden_dim))
        force_layers.append(activation())
        
        for _ in range(num_hidden_layers - 1):
            force_layers.append(nn.Linear(hidden_dim, hidden_dim))
            force_layers.append(activation())
        
        force_layers.append(nn.Linear(hidden_dim, hidden_dim))  # Output for dot product
        self.force_branch_net = nn.Sequential(*force_layers)
        
        # Build trunk network with flexible number of hidden layers
        # Output dimension should be hidden_dim for proper DeepONet
        trunk_layers = []
        trunk_layers.append(nn.Linear(trunk_input_dim, hidden_dim))
        trunk_layers.append(activation())
        
        for _ in range(num_hidden_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(activation())
        
        trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))  # Output for dot product
        self.trunk_net = nn.Sequential(*trunk_layers)

    
    def forward(self, branch_input, trunk_input):
        batch_size, num_timesteps, _ = trunk_input.shape  # Extract batch and time dims
        
        ic_input = branch_input[:, :self.ic_dim]  # First four elements are initial conditions
        force_input = branch_input[:, self.ic_dim:]  # Remaining elements are external forces

        # Branch networks output: (batch_size, hidden_dim) each
        ic_output = self.ic_branch_net(ic_input)
        force_output = self.force_branch_net(force_input)
        
        # Combine branch outputs (can use addition, multiplication, or concatenation)
        # Here using element-wise multiplication like the original, but keeping DeepONet structure
        combined_branch = ic_output * force_output  # (batch_size, hidden_dim)
        
        # Reshape to (batch_size, output_dim, hidden_dim // output_dim)
        # We need to adjust the dimensions to make this work properly
        branch_features_per_output = self.hidden_dim // self.output_dim
        if self.hidden_dim % self.output_dim != 0:
            # If not evenly divisible, we need to pad or adjust
            branch_features_per_output = self.hidden_dim // self.output_dim + 1
            # Pad the combined_branch to make it divisible
            padding_size = branch_features_per_output * self.output_dim - self.hidden_dim
            if padding_size > 0:
                combined_branch = torch.cat([combined_branch, torch.zeros(batch_size, padding_size, device=combined_branch.device)], dim=1)
        
        # Reshape to (batch_size, output_dim, branch_features_per_output)
        branch_reshaped = combined_branch.view(batch_size, self.output_dim, branch_features_per_output)
        
        # Trunk network output: (batch_size * num_timesteps, hidden_dim)
        trunk_output = self.trunk_net(trunk_input.view(-1, trunk_input.shape[-1]))
        # Take only the features we need for dot product
        trunk_output = trunk_output[:, :branch_features_per_output]
        # Reshape to (batch_size, num_timesteps, branch_features_per_output)
        trunk_output = trunk_output.view(batch_size, num_timesteps, branch_features_per_output)
        
        # DeepONet operation: dot product between branch and trunk outputs
        # branch_reshaped: (batch_size, output_dim, branch_features_per_output)
        # trunk_output: (batch_size, num_timesteps, branch_features_per_output)
        # Result: (batch_size, output_dim, num_timesteps)
        output = torch.einsum('boh,bth->bot', branch_reshaped, trunk_output)
        
        # Transpose to get (batch_size, num_timesteps, output_dim)
        output = output.transpose(1, 2)
        
        return output



# Optional: if you want 'sin' activation
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)



class DeepONetUnstackedf(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, output_dim, hidden_dim=64, num_hidden_layers=3, act_func='tanh'):
        super(DeepONetUnstackedf, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Choose activation class
        if act_func == 'tanh':
            activation = nn.Tanh
        elif act_func == 'relu':
            activation = nn.ReLU
        elif act_func == 'sin':
            activation = Sine
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")

        # Build branch network with flexible number of hidden layers
        # Output dimension should be hidden_dim * output_dim for proper DeepONet
        branch_layers = []
        branch_layers.append(nn.Linear(branch_input_dim, hidden_dim))
        branch_layers.append(activation())
        
        for _ in range(num_hidden_layers - 1):
            branch_layers.append(nn.Linear(hidden_dim, hidden_dim))
            branch_layers.append(activation())
        
        branch_layers.append(nn.Linear(hidden_dim, hidden_dim * output_dim))  # Output for dot product
        self.branch_net = nn.Sequential(*branch_layers)

        # Build trunk network with flexible number of hidden layers
        # Output dimension should be hidden_dim for proper DeepONet
        trunk_layers = []
        trunk_layers.append(nn.Linear(trunk_input_dim, hidden_dim))
        trunk_layers.append(activation())
        
        for _ in range(num_hidden_layers - 1):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(activation())
        
        trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))  # Output for dot product
        self.trunk_net = nn.Sequential(*trunk_layers)
    
    def forward(self, branch_input, trunk_input):
        batch_size, num_timesteps, _ = trunk_input.shape
        
        # Branch network output: (batch_size, hidden_dim * output_dim)
        branch_output = self.branch_net(branch_input)
        # Reshape to (batch_size, output_dim, hidden_dim)
        branch_output = branch_output.view(batch_size, self.output_dim, self.hidden_dim)
        
        # Trunk network output: (batch_size * num_timesteps, hidden_dim)
        trunk_output = self.trunk_net(trunk_input.view(-1, trunk_input.shape[-1]))
        # Reshape to (batch_size, num_timesteps, hidden_dim)
        trunk_output = trunk_output.view(batch_size, num_timesteps, self.hidden_dim)
        
        # DeepONet operation: dot product between branch and trunk outputs
        # branch_output: (batch_size, output_dim, hidden_dim)
        # trunk_output: (batch_size, trunk points, hidden_dim)
        # Result: (batch_size, trunk points)
        output = torch.einsum('boh,bth->bot', branch_output, trunk_output)

        # Transpose to get (batch_size, trunk points, output_dim)

        return output.transpose(1, 2)


class SinglePINNNet(nn.Module):
    """
    A SINGLE network that maps [branch, trunk] -> output, applied per trunk point.
    I/O:
      forward(branch_input, trunk_input) -> (B, T, O)
        branch_input: (B, Bdim)
        trunk_input:  (B, T, Tdim)
        returns:      (B, T, O)
    """
    def __init__(self, branch_input_dim, trunk_input_dim, output_dim,
                 hidden_dim=64, num_hidden_layers=4, act_func='tanh'):
        super().__init__()
                # Choose activation class
        if act_func == 'tanh':
            activation = nn.Tanh
        elif act_func == 'relu':
            activation = nn.ReLU
        elif act_func == 'sin':
            activation = Sine
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")
        in_dim = branch_input_dim + trunk_input_dim
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.hidden_dim = hidden_dim

    def forward(self, branch_input, trunk_input):
        """
        branch_input: (B, Bdim)
        trunk_input:  (B, T, Tdim)
        returns:      (B, T, O)
        """
        B, T, _ = trunk_input.shape
        # Tile branch features across trunk points
        branch_tiled = branch_input.unsqueeze(1).expand(B, T, -1)   # (B, T, Bdim)
        x = torch.cat([branch_tiled, trunk_input], dim=-1)          # (B, T, Bdim+Tdim)
        x = x.reshape(B*T, -1)                                      # (B*T, in_dim)
        y = self.mlp(x)                                             # (B*T, O)
        return y.view(B, T, -1)                                     # (B, T, O)
    



class CosineActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Network(nn.Module):
    """
    A class to represent a dynamic neural network model with dynamic number of layers based on the respective argument.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers,activation="tanh"):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        if activation=="swish":
            self.activation = SwishActivation()#CosineActivation #nn.Tanh
        elif activation=="cos":
            self.activation = CosineActivation() #nn.Tanh
        elif activation=="tanh":
            self.activation = nn.Tanh()
        self.hidden = []
        self.hidden.append(nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Forward pass of the dynamic neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        for i in range(self.num_layers):
            x = self.activation(self.hidden[i](x))
        x = self.output(x)
        return x
  