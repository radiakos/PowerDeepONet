import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from src.neural_operator.neural_operators import *
from src.neural_operator.early_stopping import EarlyStopping
import wandb
import torch.autograd.functional as func
from torch.func import jvp as func_jvp
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
import platform
import time
from src.neural_operator.operator_actions import DeepONetGPUDataset, TrajectoryDeepONetDataset, CollocationDeepONetDataset, load_hd5f, load_hd5f_efficient
from tqdm import tqdm
from pytorch_optimizer import SOAP
import json


class DeepONetActions():
    """
    A class used to define the actions of the neural network model

    Attributes
    ----------
    cfg (dict) : configuration file
    input_dim (int) : number of input features
    hidden_dim (int) : number of hidden neural network layers
    output_dim (int) : number of output features
    learning_rate (float) : learning rate of the optimizer
    model (Net) : neural network model class
    criterion (nn.Module) : loss function
    optimizer (optim) : optimizer
    scheduler (optim) : learning rate scheduler
    SM_model (SM_modelling) : class for creating the synchronous machine model
    machine_params (dict) : parameters of the synchronous machine
    system_params (dict) : parameters of the power system
    modelling_eq (CreateSolver) : class for solving the synchronous machine model
    flag_for_modelling (bool) : flag for using the synchronous machine model
    device (torch.device) : device to run the model
    
    Methods
    -------
    define_nn_model()
        This function defines the neural network model
    custom_loss(loss_name)
        This function defines the loss function
    custom_optimizer(optimizer_name, learning_rate)
        This function defines the optimizer
    custom_learning_rate(lr_name)
        This function defines the learning rate scheduler
    custom_weight_loss_updated(weight_data, weight_dt, weight_pinn, data_loss, dt_loss, pinn_loss, epoch)
        This function updates the weights of the loss functions
    weight_init(module, init_name)
        This function initializes the weights of the neural network model
    test(x_test)
        This function tests the neural network model
    plot(x_train, y_train, var=0)
        This function plots the data for a specific variable
    plot_all(x_train, y_train)
        This function plots all the data in pairs
    plot_all_dt(x_train, y_train)
        This function plots the derivative of all the data in pairs
    forward_nn(time, no_time)
        This function calculates the output of the neural network model, input is given as time and the other input columns
    forward_pass(x_train)
        This function calculates the output of the neural network model, input is given as the one whole tensor


     
    """
    def __init__(self, cfg, modelling_full, *args): # The modelling equations are used, must be predefined, more choices to be added such as dynamic modelling
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.modelling_full = modelling_full # The modelling equations are used, must be predefined, more choices to be added such as dynamic modelling

        self.model = self.define_nn_model() # Create an instance of the class Net
            # Set model precision
        if self.cfg.flag_for_precision:
            self.model = self.model.double()
        else:
            self.model = self.model.float()
        self.weight_init(self.model, cfg.nn.weight_init) # Initialize the weights of the Net
        self.model = self.model.to(self.device)

        self.criterion = self.custom_loss(cfg.nn.loss_criterion) # Define the loss function
        self.criterion_mae = nn.L1Loss() # Define the MAE loss for testing
        self.optimizer = self.custom_optimizer(cfg.nn.optimizer, cfg.nn.lr) # Define the optimizer
        self.scheduler = self.custom_learning_rate(cfg.nn.lr_scheduler) # Define the learning rate scheduler
        self.early_stopping = EarlyStopping(patience=cfg.nn.early_stopping_patience, verbose=True, delta=cfg.nn.early_stopping_min_delta)
        


    def define_nn_model(self):
        """
        This function defines the neural network model
        """
        print("Selected deep learning model: ",self.cfg.nn.type,"with ", self.cfg.nn.hidden_dim , " hidden neurons and", self.cfg.nn.hidden_layers, " hidden layers and", self.cfg.nn.activation_function, "activation function")

        if self.cfg.nn.type == "DeepONetStackedf": 
            model = DeepONetStackedf(ic_dim=4, force_dim=2000, trunk_input_dim=1, output_dim=4, hidden_dim=self.cfg.nn.hidden_dim, num_hidden_layers= self.cfg.nn.hidden_layers, act_func = self.cfg.nn.activation_function)
        elif self.cfg.nn.type == "DeepONetUnstackedf" : 
            model = DeepONetUnstackedf(branch_input_dim=2004, trunk_input_dim=1, output_dim=4, hidden_dim=self.cfg.nn.hidden_dim, num_hidden_layers= self.cfg.nn.hidden_layers, act_func = self.cfg.nn.activation_function)
        elif self.cfg.nn.type == "SinglePINN": # Dynamic architecture of the neural network
            model = SinglePINNNet(branch_input_dim=2004, trunk_input_dim=1, output_dim=4, hidden_dim=self.cfg.nn.hidden_dim, num_hidden_layers=self.cfg.nn.hidden_layers, act_func=self.cfg.nn.activation_function)
        else: 
            raise Exception("NN type not found")
        if platform.system() == "Windows":
            #print("Warning: torch.compile() is not supported on Windows. Returning uncompiled model.")
            return model  # Return model as-is
        else:
            return model # Use torch.compile() on supported OS

    def custom_loss(self, loss_name):
        """
        This function defines the loss function
        
        Args:
            loss_name (str): name of the loss function
        
        Returns:
            criterion (nn.Module): loss function
        """
        if loss_name == 'MSELoss': # Mean Squared Error Loss
            criterion = nn.MSELoss()
        elif loss_name == 'L1Loss': # Mean Absolute Error Loss
            criterion = nn.L1Loss()
        elif loss_name == 'SmoothL1Loss': # Huber Loss
            criterion = nn.SmoothL1Loss()
        else:
            raise Exception("Loss not found")
        return criterion
        
    def custom_optimizer(self, optimizer_name, learning_rate):
        """
        This function defines the optimizer

        Args:
            optimizer_name (str): name of the optimizer
            learning_rate (float): learning rate of the optimizer
        
        Returns:
            optimizer (optim): optimizer
        """
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adam_decay':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0001)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'LBFGS':
            optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')
        elif optimizer_name == 'SOAP':
            optimizer = SOAP(self.model.parameters(), lr = 3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
            #optimizer = SOAP(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        else:
            raise Exception("Optimizer not found")
        return optimizer
        
    def custom_learning_rate(self, lr_name): # Choose between "StepLR", "MultiStepLR", "ExponentialLR", "ReduceLROnPlateau
        """
        This function defines the learning rate scheduler

        Args:
            lr_name (str): name of the learning rate scheduler

        Returns:
            scheduler (optim): learning rate scheduler
        """
        if lr_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif lr_name == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,10000], gamma=0.1)
        elif lr_name == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        elif lr_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif lr_name == 'No_scheduler':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        else:
            raise Exception("Learning rate not found")
        return scheduler
        


    def weight_init(self,module, init_name):
        """
        This function initializes the weights of the neural network model
        
        Args:
            module (Net): neural network model
            init_name (str): name of the initialization method
        """
        for m in module.modules():
            if type(m) == nn.Linear:
                if init_name == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif init_name == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif init_name == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight)
                elif init_name == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                elif init_name == 'normal':
                   pass
                else:
                    raise Exception("Initialization not found")
        return
        
    # def calculate_autograd(self, model, branch_tensor, trunk_tensor):
    #     """
    #     Calculates the output of the model and its derivative w.r.t. trunk_tensor.
    #     Assumes model(branch, trunk) returns shape (batch, time, state_dim).
    #     """
    #     trunk_tensor.requires_grad_(True)  # Ensure gradients can be tracked

    #     def model_with_trunk_input(t):
    #         return model(branch_tensor, t)

    #     # Create perturbation vector v of ones with same shape as trunk_tensor
    #     v = torch.ones_like(trunk_tensor)

    #     y, dy_dt = torch.autograd.functional.jvp(
    #         model_with_trunk_input,
    #         inputs=trunk_tensor,
    #         v=v,
    #         create_graph=True
    #     )
    #     return y, dy_dt
    

    def calculate_autograd(self, model, branch_tensor, trunk_tensor, t_index=0):
        """
        Returns (y, dy_dt) where dy_dt = ∂y/∂t.
        Assumes model(branch, trunk) -> (B, T, state_dim)
        and trunk_tensor has last-dim features like [t, x, ...] with t at t_index.
        """

        # Build tangent v that is 1.0 in the t-channel, 0 elsewhere
        v = torch.zeros_like(trunk_tensor)
        v[..., t_index] = 1.0

        # Define function of trunk only (branch closed over)
        f = lambda trunk: model(branch_tensor, trunk)

        # Forward-mode JVP: y = f(trunk), dy_dt = J_f(trunk) @ v  (i.e., ∂y/∂t)
        y, dy_dt = func_jvp(f, (trunk_tensor,), (v,))

        return y, dy_dt

    def calculate_autograd2(self, model, branch_tensor, trunk_tensor):
        # ensure trunk_tensor.requires_grad is True
        #trunk_tensor = trunk_tensor.clone().detach().requires_grad_(True)

        # now func(t) passes t into the model
        func = lambda t: model(branch_tensor, t)

        # inputs and v must be tuples
        inputs = (trunk_tensor,)
        v      = (torch.ones_like(trunk_tensor),)

        y, dy_dt = torch.autograd.functional.jvp(
            func,
            inputs,
            v,
            create_graph=True,
        )
        return y, dy_dt



    def calculate_from_ode(self, modelling_full, output, branch_input):
        force_input = branch_input[:,4:]
        force_time = torch.stack([force_input[:, :1000], force_input[:, 1000:]], dim=2)
        rhs = modelling_full.odequations(0, output.unbind(dim=2), force_time.unbind(dim=2)) # calculate the right hand side of the ODE

        rhs = torch.stack(rhs, dim=2)
        return rhs

    def calculate_from_ode_new(self, modelling_full, output, branch_input, trunk_input): 
        # Split the force input into two 1000-step signals
        force_input = branch_input[:, 4:]
        force_time = torch.stack([force_input[:, :1000], force_input[:, 1000:]], dim=2)  # shape [batch, 1000, 2]

        # Compute indices from trunk_input (scaled by 1000)
        idx = (trunk_input * 1000).long() - 1  # shape [batch, n_points, 1]
        idx = idx.clamp(min=0, max=999)  # ensure valid indexing

        # Select the corresponding time steps from force_time
        selected_force = torch.gather(force_time, 1, idx.expand(-1, -1, force_time.size(-1)))  # shape [batch, n_points, 2]

        # Compute the ODE right-hand side using the selected force values
        rhs = modelling_full.odequations(0, output.unbind(dim=2), selected_force.unbind(dim=2))

        rhs = torch.stack(rhs, dim=2)
        return rhs





    def folder_name_f2(self, cfg):
        weight_data, weight_dt, weight_pinn, weight_pinn_ic = cfg.nn.weighting.weights

        self.weight_data = weight_data
        self.weight_dt = weight_dt
        self.weight_pinn = weight_pinn
        self.weight_pinn_ic = weight_pinn_ic

        if weight_data == weight_dt == 0:
            folder_name = "pinn_ic" if weight_pinn_ic > 0 else "pinn"
        elif weight_dt == weight_pinn == 0:
            folder_name = "data_ic" if weight_pinn_ic > 0 else "data"
        elif weight_data == 0:
            folder_name = "dt_pinn" if weight_pinn_ic == 0 else "pinn_ic"
        elif weight_dt > 0 and weight_pinn > 0:
            folder_name = "data_dt_pinn_ic" if weight_pinn_ic > 0 else "data_dt_pinn"
        elif weight_dt > 0 and weight_pinn == weight_pinn_ic == 0:
            folder_name = "data_dt"
        elif weight_data == 0 and weight_pinn > 0:
            folder_name = "pinn_ic" if weight_pinn_ic > 0 else "pinn"
        else:
            raise Exception("Folder name not found")

        return folder_name
    
    


        
    def test(self, x_test):
        """
        This function tests the neural network model

        Args:
            x_test (torch.Tensor): input data

        Returns:
            y_pred (torch.Tensor): predicted output data
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.forward_pass(x_test)
        return y_pred


    
    def calc_adapt_criterion_loss(self, x_train, y_train, output):
        """
        This function calculates the loss with the adaptive criterion
        """
        if self.cfg.nn.time_factored_loss == True:
            time = x_train[:,0].unsqueeze(1) # get the time column
            end_time = self.cfg.time 
            end_time = torch.tensor([end_time]).to(self.device)
            time_factor = end_time - time
            self.criterion2 = self.criterion.__class__(reduction='none')
            # Calculate element-wise L1 loss and then scale by time factor
            base_loss = self.criterion2(y_train, output) * time_factor # L1 loss
            #base_loss = base_loss  # Scale by time factor
            # calculate the mean of the loss per column
            loss_list = [base_loss[:, i].mean() for i in range(y_train.shape[1])]
            return base_loss.mean(), loss_list # Aggregate the loss
        else:
            loss_list = [self.criterion(y_train[:, i], output[:, i]) for i in range(y_train.shape[1])]
            return torch.mean(torch.stack(loss_list)), loss_list



     
    def pinn_train(self, training_data_file, training_col_data_file, training_init_data_file, validation_data_file, wandb_run=None):
        """
        This function trains the neural network model

        Args:
            num_of_skip_data_points (int): number of data points to skip
            num_of_skip_col_points (int): number of collocation points to skip
            num_of_skip_val_points (int): number of validation points to skip
            wandb_run (wandb): wandb run object
        """      
        self.wandb_run = wandb_run  
        # folder to find the datasets
        folder_path = "./"+self.cfg.dirs.dataset_dir+"/" + self.cfg.model.model_flag + '/dataset_v' + str(self.cfg.dataset.number)
        device =self.device
        # Load all data once into memory (if feasible)
        if training_data_file is not None:
            branch_input, trunk_input, target = load_hd5f_efficient(training_data_file,config=self.cfg)
            branch_input = branch_input.to(device)
            trunk_input = trunk_input.to(device) 
            target = target.to(device)
        if training_col_data_file is not None:
            branch_input_col, trunk_input_col = load_hd5f_efficient(training_col_data_file,config=self.cfg)
            branch_input_col = branch_input_col.to(device)
            trunk_input_col = trunk_input_col.to(device)
        if training_init_data_file is not None:
            branch_input_init, trunk_input_init, target_init = load_hd5f_efficient(training_init_data_file,config=self.cfg)
            branch_input_init = branch_input_init.to(device)
            trunk_input_init = trunk_input_init.to(device)
            target_init = target_init.to(device)

        branch_input_val, trunk_input_val, target_val = load_hd5f_efficient(validation_data_file,config=self.cfg)
        branch_input_val = branch_input_val.to(device)
        trunk_input_val = trunk_input_val.to(device)
        target_val = target_val.to(device)

        print("Number of labeled training data:", len(branch_input),
            "Number of collocation points:", len(branch_input_col),
            "Number of collocation points (IC):", len(trunk_input_init),
            "Number of validation data:", len(branch_input_val),''
            'dataset_v' + str(self.cfg.dataset.number))

            # Create DataLoaders with multiple workers for efficiency
        batch_size = self.cfg.nn.batch_size if self.cfg.nn.batch_size != "None" else len(branch_input)
        
        # Create DataLoaders with optimizations
        train_dataset = DeepONetGPUDataset(branch_input, trunk_input, target)
        col_dataset = DeepONetGPUDataset(branch_input_col, trunk_input_col)
        train_ic_dataset = DeepONetGPUDataset(branch_input_init, trunk_input_init, target_init)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        col_loader = DataLoader(col_dataset, batch_size=batch_size, shuffle=True)
        init_loader = DataLoader(train_ic_dataset, batch_size=batch_size, shuffle=True)
        
        
        folder_name=self.folder_name_f2(self.cfg) # folder to save the model
        os.makedirs(os.path.join(self.cfg.dirs.model_dir, folder_name),exist_ok=True)
        device = self.device

        for epoch in range(1,self.cfg.nn.num_epochs+1):
            self.model.train() # set the model to training mode
            for _ in range(len(train_loader)):  
                branch_input, trunk_input, target = next(iter(train_loader))
                branch_input_col, trunk_input_col = next(iter(col_loader))
                branch_input_init, trunk_input_init, target_init = next(iter(init_loader))


                def closure():

                    #branch_input = branch_input.to(device)
                    #trunk_input = trunk_input.to(device)
                    #target = target.to(device)


                    output, dy_dt = self.calculate_autograd(self.model, branch_input, trunk_input)       # [B, 1000, 4]
                    rhs_true = self.calculate_from_ode(self.modelling_full, target, branch_input) 

                    output_col, dy_dt_col = self.calculate_autograd(self.model, branch_input_col, trunk_input_col)  
                    rhs_col = self.calculate_from_ode(self.modelling_full, output_col, branch_input_col) 

                    output_init = self.model(branch_input_init, trunk_input_init)
                    # Data and Physics Losses for the data points

                    loss_data = self.criterion(output, target) # calculate the data loss
                    loss_dt =torch.stack([self.criterion(dy_dt[:, i], rhs_true[:, i]) for i in range(dy_dt.shape[1])])
                    mean_loss_dt = loss_dt.mean() # Stack and then mean
                    # _, _ = self.calc_adapt_criterion_loss(x_col_batch, dydt1, ode1)   #Later development
                    # Physics Loss for the collocation points
                    loss_pinn = torch.stack([self.criterion(dy_dt_col[:, i], rhs_col[:, i]) for i in range(dy_dt_col.shape[1])])
                    mean_loss_pinn = loss_pinn.mean() # Stack and then mean
                    # Initial Condition Loss for the collocation points
                    loss_pinn_ic = self.criterion(output_init, target_init)
                    
                    # Compute the total weighted loss using static weights
                    loss_total = (self.weight_data * loss_data + 
                                self.weight_dt * mean_loss_dt + 
                                self.weight_pinn * mean_loss_pinn + 
                                self.weight_pinn_ic * loss_pinn_ic)
                    self.loss_total, self.loss_data, self.loss_dt, self.loss_pinn, self.loss_pinn_ic = loss_total, loss_data, mean_loss_dt, mean_loss_pinn, loss_pinn_ic
                    self.losses = [loss_total, loss_data, mean_loss_dt, mean_loss_pinn, loss_pinn_ic]

                    self.optimizer.zero_grad() # zero the gradients
                    loss_total.backward() # backpropagate the total weighted loss
                    
                    return loss_total
                
                self.optimizer.step(closure) # update the weights of the model
                torch.cuda.empty_cache()  
                # scaler.step(self.optimizer, closure=closure)
                # scaler.update()

            """
            if self.optimizer != "LBFGS":
                self.scheduler.step()
            """

            # Validation
            self.model.eval()
            
            val_outputs, val_dydt = self.calculate_autograd(self.model, branch_input_val, trunk_input_val)   
            rhs_val = self.calculate_from_ode(self.modelling_full, val_outputs, branch_input_val) 

            loss_val_data = self.criterion(val_outputs, target_val)
            mean_loss_val_physics = torch.stack([self.criterion(val_dydt[:, i], rhs_val[:, i]) for i in range(val_dydt.shape[1])]).mean()
                
            val_loss = loss_val_data.item() 
            val_dt_loss = mean_loss_val_physics.item() 
            

            # Static weights - no dynamic updating needed
                
            if (epoch ) % 100 == 0:
                print(f'Epoch [{epoch}/{self.cfg.nn.num_epochs}], Loss: {self.loss_total.item():.4f}, Loss_data: {self.loss_data.item():.4f}, Loss_dt: {self.loss_dt.item():.4f}, Loss_pinn: {self.loss_pinn.item():.4f} , Loss_pinn_ic : {self.loss_pinn_ic.item():.4f}, Val_loss: {val_loss}, Val_dt_loss: {val_dt_loss}')
                # log some plots to wandb
                if wandb_run is not None:
                    self.log_plot(val_outputs, target_val, epoch, wandb_run, trunk_input_val)
                    
            # log all the losses for the epoch to wandb 
            save_iteration = 500 if self.cfg.nn.optimizer != "Adam" else 5000 
            if (epoch ) % save_iteration == 0:
                
                name = f"{self.cfg.model.model_flag}{self.cfg.nn.type}_{self.cfg.seed}_{self.cfg.time}_{epoch}_{len(train_dataset)}_{len(col_dataset)}_{len(branch_input_val)}_{self.weight_data}_{self.weight_dt}_{self.weight_pinn}_{self.weight_pinn_ic}_static.pth"

                self.save_model(os.path.join(folder_name, name))
            
                if wandb_run is not None:
                    log_data = {
                        "Val_loss": val_loss,
                        "Val_dt_loss": val_dt_loss,
                        "Loss": self.loss_total.item(),
                        "Loss_data": self.loss_data.item(),
                        "Loss_dt": self.loss_dt,
                        "Loss_pinn": self.loss_pinn,
                        "Loss_pinn_ic": self.loss_pinn_ic,
                        "Weight_data": self.weight_data,
                        "Weight_dt": self.weight_dt,
                        "Weight_pinn": self.weight_pinn,
                        "Weight_pinn_ic": self.weight_pinn_ic,
                        "epoch": epoch
                    }
                    wandb_run.log(log_data)


            if self.cfg.nn.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    self.early_stopping.save_checkpoint(val_loss, self.model)
                    break

            #empty the cache to avoid memory issues
            torch.cuda.empty_cache()
            # set val_outputs to None to free up memory
            val_outputs, val_dydt, rhs_val, loss_val_data, mean_loss_val_physics = None, None, None, None, None


        if self.early_stopping.early_stop == True or (epoch ) % save_iteration != 0:
            name = f"{self.cfg.model.model_flag}{self.cfg.nn.type}_{self.cfg.time}_{epoch}_{len(train_dataset)}_{len(col_dataset)}_{len(target_val)}_{self.weight_data}_{self.weight_dt}_{self.weight_pinn}_{self.weight_pinn_ic}_static.pth"
            self.save_model(os.path.join(folder_name, name))
        self.final_name = os.path.join(folder_name, f"{self.cfg.model.model_flag}{self.cfg.nn.type}_{self.cfg.time}_{epoch}_{len(train_dataset)}_{len(col_dataset)}_{len(target_val)}_{self.weight_data}_{self.weight_dt}_{self.weight_pinn}_{self.weight_pinn_ic}_static")
        return 
    
    def test_model(self, training_data_file, wandb=None, flag_time=False, batch_sizes=[1, 10, 100, 500, 1000], repeats=100):
        """
        This function tests the neural network model

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            num_epochs (int): number of epochs
        """
        run = wandb
        branch_input, trunk_input, target = load_hd5f_efficient(training_data_file,config=self.cfg)
        branch_input = branch_input.to(self.device)
        trunk_input = trunk_input.to(self.device) 
        target = target.to(self.device)
        #print(branch_input.shape, trunk_input.shape, target.shape)
        self.model.eval()

            # --- Helper: full forward in batches (to keep loss comparable across modes) ---
       

        # --- Timing/benchmark mode ---
        if flag_time:
            results = {}
            use_cuda = torch.cuda.is_available() and (self.device.type == "cuda" or "cuda" in str(self.device))

            for bs in batch_sizes:
                times = []
                for _ in range(repeats):
                    if use_cuda:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = self.model(branch_input[:bs], trunk_input[:bs])

                    if use_cuda:
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                mean_t = float(np.mean(times))
                std_t  = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0

                print(f"[Batch {bs:>5}] avg: {mean_t:.6f}s  std: {std_t:.6f}s over {repeats} runs")
                results[bs] = {"mean_s": mean_t, "std_s": std_t, "times_s": times}

                if run is not None:
                    run.log({f"time/mean_s_bs_{bs}": mean_t, f"time/std_s_bs_{bs}": std_t})
                    # Optional: log raw run times
                    for i, t in enumerate(times, 1):
                        run.log({f"time/run_s_bs_{bs}": t, "run_index": i})

        y_pred = self.model(branch_input, trunk_input)
        test_loss = self.criterion(y_pred, target)
        print("Total test trajectories",target.shape[0])
        print(f'Loss: {test_loss.item():.8f}')
        test_loss_mae = self.criterion_mae(y_pred, target)
        test_max_ae = torch.max(torch.abs(y_pred - target))
        print(f'Max AE: {test_max_ae.item():.8f}')
        print(f'MAE Loss: {test_loss_mae.item():.8f}')
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total trainable parameters',total_trainable_params)
        if run is not None:
            run.log({"Test_loss": test_loss.item() })
            run.log({"MAE Test loss": test_loss_mae.item() })
            run.log({"Max AE": test_max_ae.item() })
            run.log({"Total trainable parameters": total_trainable_params })
        self.log_plot(y_pred, target, None, run ,trunk_input) # True instead of run
        #mae, rmse = self.loss_over_time(trunk_input, target, y_pred, run)
        metrics = self.simple_metrics(y_pred, target)
        full_path = os.path.join(self.cfg.dirs.model_dir, self.final_name)
        self.save_metrics_np(metrics, full_path)
        self.save_metrics(metrics, full_path)
        return y_pred, target, branch_input, trunk_input

    def simple_test_model(self, training_data_file, wandb=None, pinn_flag=False):
        """
        This function tests the neural network model

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            num_epochs (int): number of epochs
        """
        run = wandb
        branch_input, trunk_input, target = load_hd5f_efficient(training_data_file,config=self.cfg)
        branch_input = branch_input.to(self.device)
        trunk_input = trunk_input.to(self.device) 
        target = target.to(self.device)
        #print(branch_input.shape, trunk_input.shape, target.shape)
        self.model.eval()
        if pinn_flag:
            y_pred, y_pred_dydt = self.calculate_autograd(self.model, branch_input, trunk_input)   
            rhs_ode = self.calculate_from_ode(self.modelling_full, target, branch_input) 
            test_loss = self.criterion(y_pred, target)
            mae_loss=self.criterion_mae(y_pred, target)
            pinn_loss = self.criterion(y_pred_dydt, rhs_ode)
            mae_pinn =self.criterion_mae(y_pred_dydt, rhs_ode)
            print("Total test trajectories",target.shape[0])
            print(f'Loss: {test_loss.item():.8f}')
            print(f'PINN Loss: {pinn_loss.item():.8f}')
            print(f'MAE Loss: {mae_loss.item():.8f}')
            print(f'MAE PINN Loss: {mae_pinn.item():.8f}')
            return y_pred, target, branch_input, trunk_input, y_pred_dydt, rhs_ode

        y_pred = self.model(branch_input, trunk_input)
        test_loss = self.criterion(y_pred, target)
        print("Total test trajectories",target.shape[0])
        print(f'Loss: {test_loss.item():.8f}')
        test_loss_mae = self.criterion_mae(y_pred, target)
        print(f'MAE Loss: {test_loss_mae.item():.8f}')
        
        return y_pred, target, branch_input, trunk_input

    def simple_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        # pred/target: [B, T, V]
        err = pred - target
        ae  = err.abs()
        se  = err.pow(2)

        # --- Per-variable (aggregate over batch & time)
        per_var = {
            "mae":    ae.mean(dim=(0, 1)),          # [V]
            "rmse":   se.mean(dim=(0, 1)).sqrt(),   # [V]
            "maxae":  ae.amax(dim=(0, 1)),          # [V]
            "std_ae": ae.std(dim=(0, 1)),           # [V]
        }

        # --- Overall (aggregate over all dims)
        overall = {
            "mae":    ae.mean().item(),
            "rmse":   se.mean().sqrt().item(),
            "maxae":  ae.max().item(),
            "std_ae": ae.std().item(),
        }

        # --- Per time step (aggregate over batch)
        per_time = {
            "mae":    ae.mean(dim=0),          # [T, V]
            "rmse":   se.mean(dim=0).sqrt(),   # [T, V]
            "maxae":  ae.amax(dim=0),          # [T, V]
            "std_ae": ae.std(dim=0),           # [T, V]
        }

        # --- Per time step per var (aggregate over batch)
        per_time_var = {
            "mae":    ae.mean(dim=(0, 2)),          # [T, V]
            "rmse":   se.mean(dim=(0, 2)).sqrt(),   # [T, V]
            "maxae":  ae.amax(dim=(0, 2)),          # [T, V]
            "std_ae": ae.std(dim=(0, 2)),           # [T, V]
        }

        return {"per_variable": per_var, "per_time": per_time, "overall": overall, "per_time_var": per_time_var}


    def save_metrics_np(self, metrics, path="metrics"):
        os.makedirs(path, exist_ok=True)

        # Save per-variable and per-time metrics as .npy arrays
        for group_name, group in metrics.items():
            if isinstance(group, dict):
                for k, v in group.items():
                    if isinstance(v, torch.Tensor):
                        np.save(f"{path}/{group_name}_{k}.npy", v.detach().cpu().numpy())
                    else:  # scalars (e.g. overall)
                        np.save(f"{path}/{group_name}_{k}.npy", np.array(v))
        return

    def save_metrics(self,metrics, path="metrics"):
        os.makedirs(path, exist_ok=True)
        torch.save(metrics, f"{path}/metrics.pt")  # save as tensors

        def to_plain(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, dict):
                return {k: to_plain(v) for k, v in x.items()}
            return x

        with open(f"{path}/metrics.json", "w") as f:
            json.dump(to_plain(metrics), f, indent=2)
        return
    
    def log_plot(self, output, target, epoch, run, x_test):
        #log in wandb
        starting_traj = 0
        total_traj = 5
        num_per_traj = 1000
        down_limit = starting_traj*num_per_traj
        upper_limit = (starting_traj+total_traj)*num_per_traj
        var_name = ["theta","omega(r/s)","E_q(pu)","E_d(pu)"]
        modeling_guide_path = os.path.join(self.cfg.dirs.init_conditions_dir, "modellings_guide.yaml")
        modeling_guide = OmegaConf.load(modeling_guide_path)
        #check if proposed modeling is in the modeling guide
        for model in modeling_guide:
            model_name = model.get("name")
            if model_name == self.cfg.model.model_flag:
                self.keys = model.get("keys")

        for var in range(len(self.keys)):
            plt.figure()
            plt.title(f"Trajectories {starting_traj} to {starting_traj+total_traj} for variable {self.keys[var]}")
            
            # Reshape tensors for trajectory-based plotting if needed
            if len(output.shape) == 3:  # [batch, time, variables] format
                # Reshape to [batch*time, variables] for compatibility with original indexing
                output_reshaped = output.reshape(-1, output.shape[-1])
                target_reshaped = target.reshape(-1, target.shape[-1])
                x_test_reshaped = x_test.reshape(-1, x_test.shape[-1])
                
                # plot with x axis x_test[:,0]
                plt.plot(x_test_reshaped[down_limit:upper_limit,0].detach().cpu().numpy(), 
                        output_reshaped[down_limit:upper_limit,var].detach().cpu().numpy(), label="Predicted")
                plt.plot(x_test_reshaped[down_limit:upper_limit,0].detach().cpu().numpy(), 
                        target_reshaped[down_limit:upper_limit,var].detach().cpu().numpy(), label="True")
            else:  # Original format
                # plot with x axis x_test[:,0]
                plt.plot(x_test[down_limit:upper_limit,0].detach().cpu().numpy(), 
                        output[down_limit:upper_limit,var].detach().cpu().numpy(), label="Predicted")
                plt.plot(x_test[down_limit:upper_limit,0].detach().cpu().numpy(), 
                        target[down_limit:upper_limit,var].detach().cpu().numpy(), label="True")
            
            plt.xlabel("Time(s)")
            plt.legend()
            image = wandb.Image(plt)
            if epoch is None:
                if run is not None:
                    run.log({f"Test output {var}": image})
            else:
                if run is not None:
                    run.log({f"Output {var}": image, 'epoch': epoch})
            #plt.close()
        return
    
    def loss_over_time(self, x_test, y_test, y_pred, run = None):

        unique_values = torch.unique(x_test[:,0]) # get the unique values of the time
        mae = []
        rmse = []
        for value in unique_values: # for each time step
            index = torch.where(x_test[:,0] == value) # find the indexes of the time step
            # calculate the mae and rmse for each value
            y_pred_ = y_pred[index] # keep only the points at the specific time
            y_true = y_test[index] # keep only the points at the specific time
            mae_var = []
            rmse_var = []
            for i in range(y_pred_.shape[1]):
                mae_var.append(self.criterion_mae(y_pred_[:,i], y_true[:,i]).item()) # calculate the mae for each variable
                rmse_var.append(self.criterion(y_pred_[:,i], y_true[:,i]).item()) # calculate the rmse for each variable
            mae.append((mae_var))
            rmse.append((rmse_var))
        mae = np.array(mae)
        rmse = np.array(rmse)
        mae2 = torch.abs(y_test - y_pred)  # Calculate absolute errors for each prediction
        var_name = ["theta","omega(r/s)","E_q(pu)","E_d(pu)"]
        if run is not None:
            for i in range(y_pred_.shape[1]):
                """
                plt.figure()
                plt.title(f"MAE for variable {var_name[i]} over time")
                """
                mean_mae = np.mean(mae[:,i])
                """
                plt.plot(unique_values.detach().cpu().numpy(), mae[:,i], label=f"Mean MAE: {mean_mae}")
                plt.xlabel("Time(s)")
                plt.ylabel("MAE")
                plt.legend()
                run.log({f"MAE for variable {var_name[i]} over time": wandb.Image(plt)})
                plt.close()
                plt.figure()
                plt.title(f"RMSE for variable {var_name[i]} over time")
                """
                mean_rmse = np.mean(rmse[:,i])
                """
                plt.plot(unique_values.detach().cpu().numpy(), rmse[:,i], label=f"Mean RMSE: {mean_rmse}")
                plt.xlabel("Time(s)")
                plt.ylabel("RMSE")
                plt.legend()
                run.log({f"RMSE for variable {var_name[i]} over time": wandb.Image(plt)})
                plt.close()
                """
                max_mae = torch.max(mae2[:,i])  # Find the maximum absolute error # calculate the absolute error for each prediction, to find max mae

                #log only mean values
                run.log({f"Mean MAE for variable {self.keys[i]}": mean_mae})
                run.log({f"Mean RMSE for variable {self.keys[i]}": mean_rmse})
                run.log({f"Max MAE for variable {self.keys[i]}": max_mae.item()})
            time = unique_values.detach().cpu().numpy()
            for i in range(y_pred_.shape[1]):
                for j in range(time.shape[0]):
                    #run.log({"Maw ": loss_total.item(), 'epoch': epoch})
                    # log MAE and RMSE for each variable at time  
                
                    run.log({f"MAE for variable {self.keys[i]}": mae[j,i], "Time": time[j]})
                    run.log({f"RMSE for variable {self.keys[i]}": rmse[j,i], "Time": time[j]})

        max_mae = torch.max(mae2)  # Find the maximum absolute error
        if run is not None:
            run.log({"Test Max AE": max_mae.item()})
        #save the mae and rmse
        full_path = os.path.join(self.cfg.dirs.model_dir, self.final_name)
        np.save(full_path+"_mae.npy", mae)
        np.save(full_path+"_rmse.npy", rmse)

        return mae, rmse

    def save_model(self,name):
        """
        Save model weights to the model_dir.      

        Args:
            name (str): name of the model 
        """
        #save model to the model_dir
        model_dir = self.cfg.dirs.model_dir
        #find if there is such folder in the model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir,name)
        

        model_data = {"model_state_dict":self.model.state_dict()}

        torch.save(model_data, model_path)
        
        print("Model( and tf values) saved:", model_path)
        return
    
    def load_model(self,name=None):
        """
        Load neural network model weights from the model_dir.

        Args:
            name (str): name of the model
        """
        #load model from the model_dir
        model_dir = self.cfg.dirs.model_dir
        if not os.path.exists(model_dir) or len(os.listdir(model_dir))==0:
            raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
        if name is None:
            #find first model in the model_dir
            name=os.listdir(model_dir)[0]
            if name=='.gitkeep':
                if len(os.listdir(model_dir))==1:
                    raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
                name=os.listdir(model_dir)[1]
            print("load model:",name)
        
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            print(os.path.join(model_dir,name))
            raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")

        model_data = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        return None


    
    def plot(self, x_train, y_train, var=0):
        """
        This function plots the data

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            var (int): variable to plot
        """
        y_pred = self.test(x_train)
        x_train = x_train[:,0].cpu().detach().numpy() # x is the time
        y_train = y_train[:,var].cpu().detach().numpy() 
        y_pred = y_pred[:,var].cpu().detach().numpy()
        plt.figure()
        plt.plot(x_train, y_train, 'ro', label='Original data')
        plt.plot(x_train, y_pred, 'kx-', label='Fitted line')
        plt.show()
        return
    
    def plot_all(self, x_train, y_train):
        """
        This function plots all the data in pairs

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
        """
        y_pred = self.test(x_train)
        x_train = x_train[:,0].cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        plt.figure(figsize=(10, 5))  # Create a figure with a specific size
        for i in range(y_train.shape[1]):
            plt.subplot(1, 2, i % 2 + 1)  # Create subplots, alternating between two columns
            plt.plot(x_train, y_train[:, i],  label='Original data')
            plt.plot(x_train, y_pred[:, i],  label='Fitted line')
            plt.legend()
            if i % 2 != 0:
                plt.show()  # Show the plot after every two iterations
        return
    
    def plot_all_dt(self, x_train, y_train):
        """
        This function plots the derivative of all the data in pairs
        """
        y_pred = self.test(x_train) # Predict the output data
        dt = self.calculate_from_ode(y_train,x_train)  # Calculate the derivative of the output data
        dt_pred = self.calculate_from_ode(y_pred,x_train) # Calculate the derivative of the predicted output data
        dt = dt.cpu().detach().numpy()
        dt_pred = dt_pred.cpu().detach().numpy()
        x_train = x_train[:,0].cpu().detach().numpy() # Keep only the time column

        plt.figure(figsize=(10, 5))
        for i in range(y_train.shape[1]):
            plt.subplot(1, 2, i % 2 + 1)
            plt.plot(x_train, dt[:, i],  label='Original data')
            plt.plot(x_train, dt_pred[:, i],  label='Fitted line')
            plt.legend()
            if i % 2 != 0:
                plt.show()
        return