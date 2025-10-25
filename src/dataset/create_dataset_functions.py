from functions import *
from omegaconf import OmegaConf
import os
import torch

import wandb
import pickle
import numpy as np
from pyDOE import lhs
from scipy.integrate import solve_ivp
import time
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from src.neural_operator.operator_actions import TrajectoryDeepONetDataset, CollocationDeepONetDataset, load_hd5f




class ODE_modelling():
    def __init__(self, config):
        """
        Initializes an instance of ClassName.
        Args:
            config: The configuration object containing various parameters.

        Attributes:
            config (object): The configuration object.
            modelling_method (str): The modelling method: "Ground truth" or "Collocation".
            model_flag (str): The model flag: SM, SM_AVR or SM_AVR_GOV.
            time (str): The time interval for the simulation.
            machine_num (int): The machine number that will be used.
            init_conditions (str): The initial conditions set 
            params_dir (str): The directory for parameters: machine, avr, gov and system.
            init_conditions_dir (str): The directory for initial conditions per machine modelling type: SM, SM_AVR and SM_AVR_GOV.
            dataset_dir (str): The directory for saving the dataset.
            torch (bool): The flag to use PyTorch for the model.

        Methods:
            define_machine_params: Define the parameters of the synchronous machine based on the machine_num.
            define_system_params: Define the system parameters of the synchronous machine.
            append_element: Appends an element to each state in the given value set by iterating over a range of values.
            append_element_set: Appends an element to each state in the given value set by iterating over a range of values.
            create_init_conditions_set: Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.
            create_solver: Create the solver for the synchronous machine model.
            solve_sm_model: Solves the synchronous machine model for multiple initial conditions.
            save_dataset: Create and save dataset for the model.
            load_dataset: Load the dataset.
        """
        self.config = config
        self.model_flag = config.model.model_flag
        self.time = config.time
        self.num_of_points = config.num_of_points
        self.init_condition_bounds = config.model.init_condition_bounds
        self.sampling = config.model.sampling
        self.init_conditions_dir = config.dirs.init_conditions_dir
        self.dataset_dir = config.dirs.dataset_dir
        self.torch = config.model.torch
        self.seed = None if not hasattr(config.model, 'seed') else config.model.seed
        self.load_keys()
    
    def append_element(self, value_set, Value_range, num_ranges):
        """
        Appends an element to each state in the given value set by iterating over a range of values.

        Args:
            value_set (list): The list of states to which the element will be appended.
            Value_range (tuple): The range of values from which the element will be selected.
            num_ranges (int): The number of ranges to divide the Value_range into.

        Returns:
            list: A new list of states with the element appended.

        """
        
        new_value_set = []
        for j in range(len(value_set)):
            for i in range(num_ranges):
                value = (Value_range[0] + i * (Value_range[1] - Value_range[0]) / (num_ranges - 1) if num_ranges > 1 else Value_range[0])
                new_state = value_set[j].copy()
                new_state.extend([value])
                new_value_set.append(new_state)
        return new_value_set
    
    
    def append_element_set(self, value_set, Value_range, num_ranges):
        seed = (self.seed if self.seed is not None else np.random.randint(0, 1000))
        if self.sampling=="Random":
            points = np.random.uniform(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Linear":
            points = np.linspace(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Lhs":
            points = lhs(n=1, samples=num_ranges)
        else:
            raise Exception("Sampling method not implemented")
        
        new_value_set = []
        iterations = len(value_set) if len(value_set)>1 else 1
        for j in range(iterations):
            if len(value_set)<1:
                values = (Value_range[0] + points * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                new_value_set = [values][0].tolist()
            else:
                for i in points:
                    if isinstance(i, np.ndarray):
                        i = i.item()
                    value = (Value_range[0] + i * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                    new_state = value_set[j].copy()
                    if isinstance(new_state, np.ndarray):
                        new_state = new_state.tolist()
                    new_state.extend([value])
                    new_value_set.append(new_state)
        return new_value_set

    def load_keys(self):
        modeling_guide_path = os.path.join(self.init_conditions_dir, "modellings_guide.yaml")
        modeling_guide = OmegaConf.load(modeling_guide_path)
        for model in modeling_guide:
            model_name = model.get("name")
            if model_name == self.model_flag:
                keys = model.get("keys")
                self.fullname = model.get("fullname")
                #load keys_ext if exist in model
                keys_ext = model.get("keys_ext")
                #set attr length of keys and keys_ext
                self.keys_length = len(keys)
                self.keys_ext_length = len(keys_ext) if keys_ext else 0
                #append keys_ext to keys for the initial conditions check
                if keys_ext:
                    self.keys = keys + keys_ext
                else:
                    self.keys = keys
        if not keys:
            raise ValueError(f"Model {self.model_flag} not found in modeling guide")
        return 

    def check_ic_yaml(self,init_conditions):
        """
        Modeling guide contains all the variables that can be used in the modeling.
        Check if the variables in the initial conditions are in the modeling guide in the correct order
        """
        

        for i in range(len(init_conditions)):
            name = init_conditions[i].get("name")
            if name not in self.keys:
                raise ValueError(f"Variable {name} not found in modeling guide")
            if name != self.keys[i]: #same order as in the modeling guide
                raise ValueError(f"Variable {name} does not match the modeling variable {self.keys[i]}")
            #check if iterations are always a number:
            iterations = init_conditions[i].get("iterations")
            if not isinstance(iterations, int):
                raise ValueError(f"Variable {name} iterations must be an integer")
        return

    def create_init_conditions_set3(self):
        """
        Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.

        Returns:
            list: A matrix with all the possible combinations of initial conditions.
        """
        if self.torch:# if using torch then use the nn_init_cond.yaml file to create collocation points init conditions
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"nn_init_cond"+str(self.init_condition_bounds)+".yaml")
        else: # 
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"init_cond"+str(self.init_condition_bounds)+".yaml")
        init_conditions = OmegaConf.load(init_conditions_path)
        self.check_ic_yaml(init_conditions)

        for i in range(len(init_conditions)):
            if len(init_conditions[i]["range"]) == 1: # if unique value then set iterations to 1
                init_conditions[i]["iterations"] = 1 

        # Initialize the set_of_values and iterations lists and variables
        set_of_values = []
        iterations = []
        variables = []
        # Extract values, iterations and variables from init_conditions
        for condition in init_conditions:
            set_of_values.append(condition['range'])
            iterations.append(condition['iterations'])
            variables.append(condition['name'])

        # Calculate the number of different initial conditions
        number_of_conditions = 1
        for it in iterations:
            number_of_conditions *= it
        """
        if self.torch:
            print("Number of different initial conditions for collocation points: ", number_of_conditions)
        else:
            print("Number of different initial conditions: ", number_of_conditions)

        print(variables, "Variables")
        print(set_of_values,"Set of values for init conditions")
        print(iterations,"Iterations per value")
        """
        init_condition_table = []   
        for k in range(len(set_of_values)):
            init_condition_table = self.append_element_set(init_condition_table, set_of_values[k], iterations[k])
        return init_condition_table
    

    def create_init_conditions_set4(self, total_samples=1000, collocation_points=False):
        """
        Create initial conditions by sampling each variable independently within its range.
        Instead of combinatorial sampling, generates a fixed number of samples where each
        variable is sampled independently from its specified range.

        Args:
            total_samples (int): Total number of initial condition samples to generate.

        Returns:
            list: A matrix with initial conditions where each row is one sample.
        """
        if collocation_points:# if using collocation_points then use the nn_init_cond.yaml file to create collocation points init conditions
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"nn_init_cond"+str(self.init_condition_bounds)+".yaml")
        else: # 
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"init_cond"+str(self.init_condition_bounds)+".yaml")
        init_conditions = OmegaConf.load(init_conditions_path)
        self.check_ic_yaml(init_conditions)

        # Extract ranges and variables from init_conditions
        ranges = []
        variables = []
        for condition in init_conditions:
            ranges.append(condition['range'])
            variables.append(condition['name'])

        num_variables = len(ranges)
        
        print(f"Creating {total_samples} initial condition samples")
        print(f"Variables: {variables}")
        print(f"Ranges: {ranges}")

        # Set random seed for reproducibility if specified
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate samples based on sampling method
        if self.sampling == "Random":
            # Generate random samples for all variables at once
            samples = np.random.uniform(0, 1, (total_samples, num_variables))
        elif self.sampling == "Linear":
            # For linear sampling with multiple variables, we can use a grid approach
            # but sample points along the diagonal or use sobol sequences
            if num_variables == 1:
                samples = np.linspace(0, 1, total_samples).reshape(-1, 1)
            else:
                # Use a simple approach: sample each variable linearly but offset
                samples = np.zeros((total_samples, num_variables))
                for i in range(num_variables):
                    offset = i / num_variables
                    linear_samples = (np.linspace(0, 1, total_samples) + offset) % 1.0
                    samples[:, i] = linear_samples
        elif self.sampling == "Lhs":
            # Latin Hypercube Sampling - ideal for this type of sampling
            samples = lhs(n=num_variables, samples=total_samples)
        else:
            raise Exception(f"Sampling method {self.sampling} not implemented")

        # Convert normalized samples to actual values within ranges
        init_condition_table = []
        for sample in samples:
            condition = []
            for i, (norm_val, value_range) in enumerate(zip(sample, ranges)):
                if len(value_range) == 1:
                    # Single value range
                    actual_val = value_range[0]
                else:
                    # Map from [0,1] to [min, max]
                    actual_val = value_range[0] + norm_val * (value_range[1] - value_range[0])
                condition.append(actual_val)
            init_condition_table.append(condition)

        print(f"Generated {len(init_condition_table)} initial condition samples using {self.sampling} sampling")
        return init_condition_table

    def find_steady_state(self, initial_guess, modelling_full, Vs_function, t_steady=None, method='optimization'):
        """
        Find the steady state (equilibrium point) for all variables using constant external inputs.
        
        Args:
            initial_guess (list): Initial guess for the steady state values
            modelling_full: The model containing odequations method
            Vs_function: External input function that should return CONSTANT values for steady state analysis
            t_steady (float): Time point to evaluate steady state at (default: middle of time span)
            method (str): Method to find steady state - 'optimization' or 'simulation'
            
        Returns:
            tuple: (steady_state_values, converged_flag)
                - steady_state_values (list): Steady state values for all variables
                - converged_flag (bool): True if steady state was found successfully
        """
        from scipy.optimize import fsolve, minimize
        
        # Set up time span if not provided
        if not hasattr(self, 't_span') or not hasattr(self, 't_eval'):
            self.t_span, self.t_eval = set_time(self.time, self.num_of_points)
        
        if t_steady is None:
            t_steady = (self.t_span[0] + self.t_span[1]) / 2  # Middle of time span
        
        if method == 'optimization':
            # Method 1: Use optimization to find where derivatives = 0
            def steady_state_equations(x):
                """Function that should be zero at steady state (dx/dt = 0)"""
                try:
                    # For steady state analysis, external input should be constant
                    # Vs_function should return the same values regardless of time
                    derivatives = modelling_full.odequations(1000, x, Vs_function)
                    return np.array(derivatives)
                except Exception as e:
                    print(f"Error in steady state calculation: {e}")
                    return np.ones_like(x) * 1e6  # Return large values on error
            
            # Solve for steady state using fsolve
            try:
                steady_state = fsolve(steady_state_equations, initial_guess, xtol=1e-8)
                
                final_derivatives = steady_state_equations(steady_state)
                max_derivative = np.max(np.abs(final_derivatives))
                
                # More stringent tolerance for steady state
                if max_derivative < 1e-6:
                    flag = True
                    #print(f"Steady state found! Max derivative magnitude: {max_derivative:.2e}")
                    return steady_state.tolist(), flag
                else:
                    flag = False
                    #print(f"Steady state convergence questionable. Max derivative: {max_derivative:.2e}")
                    return steady_state.tolist(), flag
                    
            except Exception as e:
                print(f"Optimization method failed: {e}")
                # Return initial guess with failed flag
                return initial_guess, False
        
        if method == 'simulation':
            # Method 2: Simulate for a long time until system reaches steady state
            print("Using simulation method to find steady state...")
            
            try:
                # Use much longer time for steady state convergence
                long_time = self.t_span[1] * 5  # 5x longer simulation
                long_time_span = (self.t_span[0], long_time)
                
                # Use more time points for better resolution
                num_points = max(self.num_of_points * 3, 3000)
                long_t_eval = np.linspace(*long_time_span, num_points)
                
                # Run long simulation with relaxed tolerances initially
                print(f"Running simulation from t={long_time_span[0]} to t={long_time_span[1]}")
                solution = solve_ivp(
                    modelling_full.odequations, 
                    long_time_span, 
                    initial_guess, 
                    t_eval=long_t_eval, 
                    args=(Vs_function,), 
                    method='RK45',
                    rtol=1e-6, 
                    atol=1e-8,
                    max_step=0.1
                )
                
                if solution.success:
                    # Take the final values as steady state
                    steady_state = solution.y[:, -1]
                    
                    # Check if system has actually reached steady state
                    # by comparing last 20% of simulation
                    last_20_percent = int(0.8 * len(solution.t))
                    final_values = solution.y[:, last_20_percent:]
                    
                    if final_values.shape[1] > 1:  # Make sure we have enough points
                        # Calculate variation in the last 20% of simulation
                        variations = np.std(final_values, axis=1) / (np.abs(np.mean(final_values, axis=1)) + 1e-12)
                        max_variation = np.max(variations)
                        
                        if max_variation < 0.05:  # Less than 5% variation
                            flag = True
                            print(f"Steady state reached via simulation! Max variation: {max_variation:.4f}")
                            return steady_state.tolist(), flag
                        else:
                            flag = False
                            print(f"System may not have reached steady state. Max variation: {max_variation:.4f}")
                            return steady_state.tolist(), flag
                    else:
                        print("Not enough simulation points to assess steady state convergence")
                        return steady_state.tolist(), False
                else:
                    print(f"Simulation failed: {solution.message}")
                    return initial_guess, False
                    
            except Exception as e:
                print(f"Simulation method encountered error: {e}")
                return initial_guess, False
        
        return initial_guess, False

    def solve(self, x0, modelling_full):
        """
        Solve the differential equations for the synchronous machine model.

        Parameters:
        - x0: list of initial state variables

        Returns:
        - solution: solution of the differential equations
        """

        
        solution = solve_ivp(modelling_full.odequations, self.t_span, x0, t_eval=self.t_eval)
        return solution

    def solve_sm_model(self, init_conditions, modelling_full, Vs_function, flag_create_dataset, flag_time=False, steady_flag = False):
        """
        Solves the synchronous machine model for multiple initial conditions using enhanced ExternalInput.

        Args:
            init_conditions (list): List of initial conditions for which the model needs to be solved.
            modelling_full: The model containing odequations method.
            Vs_function: Enhanced ExternalInput instance with automatic parameter updates and config swapping.
            flag_create_dataset (bool): Whether to create solution arrays.
            flag_time (bool): Whether to track timing information.

        Returns:
            tuple: (time_all, solution_all, external_all)
                - time_all: List of time arrays for each trajectory
                - solution_all: List of solution arrays for each trajectory  
                - external_all: List of external input (Vs, theta) tuples

        """
        self.t_span, self.t_eval = set_time(self.time, self.num_of_points)
        solution_all = []
        external_all = []
        time_all = []
        
        if steady_flag:
            steady_state_sim_all = []
            steady_state_opt_all = []
            
        # Check if this is the enhanced ExternalInput or legacy version
        is_enhanced_input = hasattr(Vs_function, 'force_new_trajectory') and hasattr(Vs_function, 'dual_system')
        if flag_time:
            time_list = [] 
        for i in range(len(init_conditions)):
            # Handle different types of ExternalInput
            if is_enhanced_input:
                # Enhanced ExternalInput: Force new trajectory with parameter updates
                if Vs_function.mode == "dual_ode":
                    Vs_function.force_new_trajectory(
                        swap_config=True,  # Allow config swapping for diversity
                        swap_mode='random'  # Random config selection
                    )
                # For other modes, the ExternalInput will handle updates internally
            else:
                # Legacy ExternalInput: Use old randomize_spec method
                if hasattr(Vs_function, 'randomize_spec'):
                    Vs_function.randomize_spec()
            
            if flag_create_dataset == 'labelled':
                if flag_time:
                    start_time = time.time()   
                    start_per_iteration = start_time
                    
                solution = solve_ivp(modelling_full.odequations, self.t_span, init_conditions[i], 
                                   t_eval=self.t_eval, args=(Vs_function,))
                
                if steady_flag:
                    steady_state_sim = 0
                    #steady_state_sim = self.find_steady_state(init_conditions[i], modelling_full, Vs_function, method='simulation')
                    
                    # Create a constant external input function using steady state reference values
                    try:
                        # Get steady state reference values (voltage_ref, theta_ref)
                        voltage_ref, theta_ref = Vs_function.steady()
                        
                        # Create constant external input function
                        def constant_external_input(t):
                            """Constant external input for steady state analysis"""
                            if hasattr(t, '__len__'):
                                # If t is array-like, return arrays of constant values
                                return (np.full_like(t, voltage_ref), np.full_like(t, theta_ref))
                            else:
                                # If t is scalar, return scalar values
                                return (voltage_ref, theta_ref)
                        
                        # Find steady state with constant external input
                        steady_state_opt = self.find_steady_state(init_conditions[i], modelling_full, constant_external_input, method='optimization')
                        
                    except Exception as e:
                        print(f"Error getting steady state reference values: {e}")
                        # Fallback: use default reference values
                        def constant_external_input(t):
                            if hasattr(t, '__len__'):
                                return (np.ones_like(t), np.zeros_like(t))  # Default: V=1.0, θ=0.0
                            else:
                                return (1.0, 0.0)
                        steady_state_opt = self.find_steady_state(init_conditions[i], modelling_full, constant_external_input, method='optimization')
                    if steady_flag:
                        steady_state_sim_all.append(steady_state_sim)
                        steady_state_opt_all.append(steady_state_opt)
                if flag_time:
                    end_per_iteration = time.time()
                    time_list.append(end_per_iteration - start_per_iteration)
                    start_per_iteration = end_per_iteration
                solution_all.append(solution.y)
                time_all.append(solution.t)
                
                # Get external input values at solution time points
                Vs_vals, theta_vals = zip(*((float(v), float(th)) for v, th in (Vs_function(t) for t in solution.t)))
            else:
                time_all.append(self.t_eval)
                # Get external input values at evaluation time points
                Vs_vals, theta_vals = zip(*((float(v), float(th)) for v, th in (Vs_function(t) for t in self.t_eval)))
            
            # Store external input data
            external_all.append((Vs_vals, theta_vals))
            
            
            
        if flag_time:
            print("Mean time per iteration: ", np.mean(time_list), " and std: ", np.std(time_list))
            print(f"Time taken to solve the model for {len(init_conditions)} initial conditions: {np.sum(time_list)} seconds.")
        if steady_flag: 
            return time_all, solution_all, external_all ,steady_state_sim_all, steady_state_opt_all
        return time_all, solution_all, external_all


    def create_dataset(self, init_conditions, time_all, solution, external_all, flag_create_dataset):
        """
        Create dataset for the model.

        Args:
            init_conditions (list): List of initial conditions.
            time_all (list): List of time arrays.
            solution (list): The solution of the differential equations of the synchronous machine.
            external_all (list): List of external input (Vs, theta) tuples.
            trajectory_info_all (list): Optional trajectory information from enhanced ExternalInput (not saved).
            flag_create_dataset (bool): Whether solution data was created.

        Returns:
            list: The dataset of the synchronous machine (simple format).
        """
        dataset = []
        if flag_create_dataset == 'initial_conditions':
            for i in range(len(init_conditions)):
                r = [0]  # start with an empty list for each trajectory
                r.append(init_conditions[i])  # append the initial conditions to the directory
                for j in range(len(external_all[i])):
                    r.append(external_all[i][j])
                r.append(init_conditions[i])  # append the solution at the initial time step
                dataset.append(r)
            return dataset
        for i in range(len(init_conditions)):
            r = [time_all[i]]  # append time to directory
            r.append(init_conditions[i])  # append the initial conditions to the directory
            for j in range(len(external_all[i])):
                r.append(external_all[i][j])
            if  flag_create_dataset == 'labelled':
                for j in range(len(solution[i])): #must save all the keys due to the preprocessing of the NN dataset
                    r.append(solution[i][j])  # append the solution at each time step
            dataset.append(r)

        return dataset
    
    
    
    def save_dataset(self, dataset):
        """
        Save the dataset in simple format.

        Args:
            dataset (list): The dataset to save.

        Returns:
            str: The path where the dataset was saved.
        """
        # check if folder exists if not create it
        if not os.path.exists(os.path.join(self.dataset_dir, self.model_flag)):
            os.makedirs(os.path.join(self.dataset_dir, self.model_flag))

        # count the number of files in the directory
        num_files = len([f for f in os.listdir(os.path.join(self.dataset_dir, self.model_flag)) 
                        if os.path.isfile(os.path.join(self.dataset_dir, self.model_flag, f))])
        
        dataset_name = f"dataset_v{num_files + 1}"
        dataset_path = os.path.join(self.dataset_dir, self.model_flag, dataset_name + ".pkl")
        
        print(f"Number of files in the directory: {num_files}")
        print(f'Saving dataset "{self.model_flag}/{dataset_name}" with {len(dataset)} trajectories')
        
        # save the dataset as pickle in the dataset directory
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
            
        print(f"Dataset saved successfully to: {dataset_path}")
        return dataset_path 

   

    def load_dataset(self, name):
        """
        Load the dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            list: The dataset of the synchronous machine.
        """
        dataset_path = os.path.join(self.dataset_dir, self.model_flag, name)
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

  
def _find_steady_state_worker(initial_guess, modelling_full, constant_external_input, t_span, num_of_points):
    """
    Global helper function for finding steady state in worker processes.
    Simplified version that doesn't rely on class methods.
    """
    from scipy.optimize import fsolve
    
    def steady_state_equations(x):
        """Function that should be zero at steady state (dx/dt = 0)"""
        try:
            derivatives = modelling_full.odequations(1000, x, constant_external_input)
            return np.array(derivatives)
        except Exception as e:
            print(f"Error in steady state calculation: {e}")
            return np.ones_like(x) * 1e6  # Return large values on error
    
    # Solve for steady state using fsolve
    try:
        steady_state = fsolve(steady_state_equations, initial_guess, xtol=1e-8)
        
        final_derivatives = steady_state_equations(steady_state)
        max_derivative = np.max(np.abs(final_derivatives))
        
        # More stringent tolerance for steady state
        if max_derivative < 1e-6:
            flag = True
            return steady_state.tolist(), flag
        else:
            flag = False
            return steady_state.tolist(), flag
            
    except Exception as e:
        print(f"Optimization method failed: {e}")
        return initial_guess, False

def gen_dataset(SM_model, modelling_full, enhanced_ext_input, flag_create_dataset, limit, dataset_name="", save_flag = True):
    init_conditions = SM_model.create_init_conditions_set4(total_samples=limit)
    
    print(f"Creating {dataset_name} dataset - Number of initial conditions {'for collocation points' if flag_create_dataset=='labelled' else 'for labeled points'}: {len(init_conditions)}")

    if 'test' in dataset_name:
        flag_time1 = True
    else:
        flag_time1 = False
    time_all, solution, ext_series = SM_model.solve_sm_model(init_conditions, modelling_full, enhanced_ext_input, flag_create_dataset=flag_create_dataset, flag_time=flag_time1)

    # Create the actual dataset
    dataset = SM_model.create_dataset(init_conditions, time_all, solution, ext_series, flag_create_dataset=flag_create_dataset)
    # Create the DeepONet dataset
    if 'labelled' in flag_create_dataset:
        deeponet_dataset = TrajectoryDeepONetDataset(dataset, SM_model.config)
        
    elif 'collocation' in flag_create_dataset:
        deeponet_dataset = CollocationDeepONetDataset(dataset, SM_model.config)

    if save_flag:
        deeponet_dataset.save_hd5f(dataset_name)
        return dataset_name
    return deeponet_dataset