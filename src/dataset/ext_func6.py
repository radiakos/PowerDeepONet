import os
import random
import pickle
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



class ODESystem:
    """
    Flexible ODE system (1st/2nd order) with randomized params/ICs.
    Small perf tweaks: RNG, bound RHS, tuples in RHS, exposed tolerances.
    """

    def __init__(self, ode_type='first_order', param_ranges=None, y0_ranges=None,
                 name="unnamed", rng=None):
        """
        Parameters:
        - ode_type: 'first_order' or 'second_order'
        - param_ranges: dict of parameter ranges
        - y0_ranges: dict of initial condition ranges
        - name: identifier for this ODE system
        - rng: numpy.random.Generator for reproducibility
        """
        if ode_type not in ('first_order', 'second_order'):
            raise ValueError("ode_type must be 'first_order' or 'second_order'")

        self.ode_type = ode_type
        self.name = name
        self.param_ranges = param_ranges or self._get_default_param_ranges()
        self.y0_ranges = y0_ranges or self._get_default_y0_ranges()
        self.rng = rng if rng is not None else np.random.default_rng()

        # Storage for current parameters and initial conditions
        self.params = None
        self.y0 = None

        self._generate_params()
        self._generate_y0()

        # Bind RHS once to avoid branching inside solver loop
        self._ode_fun = self._first_order_ode if self.ode_type == 'first_order' else self._second_order_ode

    def _get_default_param_ranges(self):
        if self.ode_type == 'first_order':
            return {
                'ref_value': (0.8, 1.2),
                'k': (1.0, 3.0),
                'noise_amplitude': (0.0, 0.05),
                'omega_noise': (0.5, 3.0)
            }
        else:  # second_order
            return {
                'ref_value': (0.9, 1.1),
                'zeta': (0.1, 0.5),
                'omega_n': (5, 15)
            }

    def _get_default_y0_ranges(self):
        if self.ode_type == 'first_order':
            return {
                'initial_value': (0.2, 0.8)
            }
        else:  # second_order
            return {
                'initial_value': (0.3, 0.7),
                'initial_derivative': (-0.5, 0.5)
            }

    def _generate_params(self):
        self.params = {k: self.rng.uniform(lo, hi) for k, (lo, hi) in self.param_ranges.items()}

    def _generate_y0(self):
        if self.ode_type == 'first_order':
            self.y0 = (self.rng.uniform(*self.y0_ranges['initial_value']),)
        else:
            self.y0 = (
                self.rng.uniform(*self.y0_ranges['initial_value']),
                self.rng.uniform(*self.y0_ranges['initial_derivative'])
            )

    def reset_params(self):
        """Reset parameters and initial conditions to new random values."""
        self._generate_params()
        self._generate_y0()

    # --------- ODE RHS (return tuples) ---------

    def _first_order_ode(self, t, y):

        """
        dy/dt = -k * (y - ref) + noise_amplitude * sin(omega_noise * t)
        """
        p = self.params
        return (-p['k'] * (y[0] - p['ref_value']) + p['noise_amplitude'] * np.sin(p['omega_noise'] * t),)

    def _second_order_ode(self, t, y):
        """
        d²y/dt² + 2*zeta*omega_n*dy/dt + omega_n²*(y - ref) = 0
        """
        p = self.params
        val, dval_dt = y
        d2 = -2 * p['zeta'] * p['omega_n'] * dval_dt - (p['omega_n'] ** 2) * (val - p['ref_value'])
        return (dval_dt, d2)

    # --------- Solver ---------

    def solve(self, t_span, t_eval=None, num_points=1000, state_index=0, rtol=1e-6, atol=1e-9, method='RK45'):
        """
        Solve the ODE system.

        Returns:
        - t: time array
        - solution: solution array for the specified state
        """
        if t_eval is None:
            t_eval = np.linspace(*t_span, num_points)
        sol = solve_ivp(
            self._ode_fun, t_span, self.y0, t_eval=t_eval,
            rtol=rtol, atol=atol, method=method, dense_output=False
        )
        return sol.t, sol.y[state_index]

    def get_info(self):
        """Get current parameter and initial condition information."""
        return {
            'name': self.name,
            'type': self.ode_type,
            'params': self.params,
            'y0': self.y0
        }


class DualODESystem:
    """
    Enhanced dual ODE system that manages multiple configurations and can swap between them.
    Each time a trajectory is sampled, parameters are automatically updated.
    """

    def __init__(self, configurations=None, rng=None):
        """
        Parameters:
        - configurations: list of dicts, each containing 'voltage_config' and 'theta_config'
        - rng: numpy.random.Generator (shared)
        """
        self.rng = rng or np.random.default_rng()
        
        # Set up configurations
        if configurations is None:
            configurations = self._get_default_configurations()
        
        self.configurations = configurations
        self.current_config_index = 0
        self.trajectory_count = 0
        
        # Initialize with first configuration
        self._initialize_ode_systems()

    def _get_default_configurations(self):
        """Get default dual system configurations."""
        return [
            {
                'name': 'fast_voltage_slow_theta',
                'voltage_config': {
                    'type': 'second_order',
                    'param_ranges': {
                        'ref_value': (0.9, 1.1),
                        'zeta': (0.6, 0.8),      # High damping = fast settling
                        'omega_n': (8, 12)       # High frequency = fast response
                    },
                    'y0_ranges': {
                        'initial_value': (0.8, 1.2),
                        'initial_derivative': (-0.1, 0.1)
                    }
                },
                'theta_config': {
                    'type': 'first_order',
                    'param_ranges': {
                        'ref_value': (0.0, 0.0),
                        'k': (0.5, 1.0),         # Low k = slow response
                        'noise_amplitude': (0.0, 0.05),
                        'omega_noise': (0.1, 0.5)
                    },
                    'y0_ranges': {
                        'initial_value': (-0.2, 0.2)
                    }
                }
            },
            {
                'name': 'slow_voltage_fast_theta',
                'voltage_config': {
                    'type': 'first_order',
                    'param_ranges': {
                        'ref_value': (0.9, 1.1),
                        'k': (1.0, 2.0),         # Moderate k = slower than theta
                        'noise_amplitude': (0.0, 0.03),
                        'omega_noise': (0.5, 1.5)
                    },
                    'y0_ranges': {
                        'initial_value': (0.7, 1.3)
                    }
                },
                'theta_config': {
                    'type': 'second_order',
                    'param_ranges': {
                        'ref_value': (0.0, 0.0),
                        'zeta': (0.2, 0.4),      # Low damping = oscillatory
                        'omega_n': (10, 18)      # High frequency = fast oscillations
                    },
                    'y0_ranges': {
                        'initial_value': (-0.3, 0.3),
                        'initial_derivative': (-0.8, 0.8)
                    }
                }
            }
        ]

    def _initialize_ode_systems(self):
        """Initialize ODE systems with current configuration."""
        config = self.configurations[self.current_config_index]
        
        self.voltage_ode = ODESystem(
            ode_type=config['voltage_config']['type'],
            param_ranges=config['voltage_config']['param_ranges'],
            y0_ranges=config['voltage_config']['y0_ranges'],
            name="voltage",
            rng=self.rng
        )
        
        self.theta_ode = ODESystem(
            ode_type=config['theta_config']['type'],
            param_ranges=config['theta_config']['param_ranges'],
            y0_ranges=config['theta_config']['y0_ranges'],
            name="theta",
            rng=self.rng
        )
        
        self.config_name = config.get('name', f'config_{self.current_config_index}')

    def swap_config(self, config_index=None, mode='sequential'):
        """
        Swap to a different configuration.
        
        Parameters:
        - config_index: specific index to swap to, or None for automatic selection
        - mode: 'sequential', 'random', or 'specific'
        """
        if config_index is None:
            if mode == 'sequential':
                self.current_config_index = (self.current_config_index + 1) % len(self.configurations)
            elif mode == 'random':
                self.current_config_index = self.rng.integers(0, len(self.configurations))
            else:
                raise ValueError("mode must be 'sequential' or 'random' when config_index is None")
        else:
            if 0 <= config_index < len(self.configurations):
                self.current_config_index = config_index
            else:
                raise ValueError(f"config_index {config_index} out of range [0, {len(self.configurations)-1}]")
        
        # Reinitialize ODE systems with new configuration
        self._initialize_ode_systems()
        
        return self.current_config_index, self.config_name

    def update_params_for_trajectory(self):
        """
        Update parameters for a new trajectory sampling.
        This is called automatically before each trajectory generation.
        """
        self.trajectory_count += 1
        
        # Reset parameters (generate new random values within ranges)
        self.voltage_ode.reset_params()
        self.theta_ode.reset_params()
        
        return self.trajectory_count

    def solve_with_param_update(self, t_span, num_points=1000, swap_config=False, 
                               swap_mode='sequential', **solve_kw):
        """
        Solve with automatic parameter update and optional config swapping.
        
        Parameters:
        - t_span: time span for integration
        - num_points: number of points in solution
        - swap_config: if True, swap configuration before solving
        - swap_mode: 'sequential' or 'random' config swapping
        - **solve_kw: additional solver parameters
        
        Returns:
        - t: time array
        - voltage: voltage solution  
        - theta: theta solution
        - trajectory_info: dict with trajectory and config information
        """
        # Optionally swap configuration
        if swap_config:
            self.swap_config(mode=swap_mode)
        
        # Always update parameters for new trajectory
        traj_count = self.update_params_for_trajectory()
        
        # Solve both ODEs
        t_eval = np.linspace(*t_span, num_points)
        _, voltage = self.voltage_ode.solve(t_span, t_eval, **solve_kw)
        _, theta = self.theta_ode.solve(t_span, t_eval, **solve_kw)
        
        # Gather trajectory information
        trajectory_info = {
            'trajectory_number': traj_count,
            'config_index': self.current_config_index,
            'config_name': self.config_name,
            'voltage_info': self.voltage_ode.get_info(),
            'theta_info': self.theta_ode.get_info()
        }
        
        return t_eval, voltage, theta, trajectory_info

    def solve(self, t_span, num_points=1000, **solve_kw):
        """
        Standard solve method (backward compatibility).
        Does NOT update parameters automatically.
        """
        t_eval = np.linspace(*t_span, num_points)
        _, voltage = self.voltage_ode.solve(t_span, t_eval, **solve_kw)
        _, theta = self.theta_ode.solve(t_span, t_eval, **solve_kw)
        return t_eval, voltage, theta

    def reset_params(self, reset_voltage=True, reset_theta=True):
        """Reset parameters for one or both ODEs."""
        if reset_voltage:
            self.voltage_ode.reset_params()
        if reset_theta:
            self.theta_ode.reset_params()

    def get_info(self):
        """Get comprehensive information about the dual system."""
        return {
            'trajectory_count': self.trajectory_count,
            'current_config_index': self.current_config_index,
            'config_name': self.config_name,
            'total_configs': len(self.configurations),
            'voltage': self.voltage_ode.get_info(),
            'theta': self.theta_ode.get_info()
        }

    def get_all_config_names(self):
        """Get names of all available configurations."""
        return [config.get('name', f'config_{i}') for i, config in enumerate(self.configurations)]

    def print_info(self):
        """Print current configuration and parameters."""
        info = self.get_info()
        print("=== Enhanced Dual ODE System Status ===")
        print(f"Trajectory Count: {info['trajectory_count']}")
        print(f"Current Config: {info['config_name']} (#{info['current_config_index']})")
        print(f"Available Configs: {self.get_all_config_names()}")

        print(f"\nVoltage ODE ({info['voltage']['type']}):")
        print(f"  Parameters: {info['voltage']['params']}")
        print(f"  Initial conditions: {info['voltage']['y0']}")

        print(f"\nTheta ODE ({info['theta']['type']}):")
        print(f"  Parameters: {info['theta']['params']}")
        print(f"  Initial conditions: {info['theta']['y0']}")


# ========================= INPUT WRAPPERS =========================

class ExternalInput:
    """
    Enhanced ExternalInput that can operate in multiple modes with automatic parameter updates:
    - 'dataset': use a precomputed dataset with cached interpolators
    - 'dual_ode': generate trajectories from one dual ODE system with config swapping
    - 'random_ode': generate trajectories with random ODE orders (legacy mode)
    """

    def __init__(self, mode="dual_ode", dataset_path=None, configurations=None,
                 t_min=None, t_max=None, num_of_points=None, rng=None,
                 auto_param_update=True, auto_config_swap=False, swap_mode='sequential'):
        """
        Parameters:
        - mode: 'dataset', 'dual_ode', or 'random_ode'
        - configurations: list of voltage/theta configuration pairs for dual_ode mode
        - auto_param_update: if True, automatically update parameters on each sample
        - auto_config_swap: if True, automatically swap configurations
        - swap_mode: 'sequential' or 'random' for automatic config swapping
        """
        self.mode = mode
        self.t_min = t_min or 0
        self.t_max = t_max or 5
        self.num_of_points = num_of_points or 1000
        self.rng = rng or np.random.default_rng()
        self.auto_param_update = auto_param_update
        self.auto_config_swap = auto_config_swap
        self.swap_mode = swap_mode

        if mode == "dataset":
            self._setup_dataset_mode(dataset_path)
        elif mode == "dual_ode":
            self._setup_dual_ode_mode(configurations)
        elif mode == "random_ode":
            self._setup_random_ode_mode()
        else:
            raise ValueError("mode must be 'dataset', 'dual_ode', or 'random_ode'.")

    def _setup_dataset_mode(self, dataset_path):
        """Set up dataset mode with cached interpolators."""
        if not dataset_path or not os.path.exists(dataset_path):
            raise ValueError("Provide valid dataset_path for 'dataset' mode.")
        with open(dataset_path, 'rb') as f:
            ds = pickle.load(f)
        if not all(k in ds for k in ("t", "Vs", "Vs_theta")):
            raise ValueError("Dataset must contain 't','Vs','Vs_theta'.")
        self.dataset = ds

        # Cache interpolators for all series once
        t = np.asarray(ds['t'])
        self._dataset_funcs = [
            (
                interp1d(t, np.asarray(v), kind='linear', fill_value='extrapolate', assume_sorted=True),
                interp1d(t, np.asarray(th), kind='linear', fill_value='extrapolate', assume_sorted=True)
            )
            for v, th in zip(ds['Vs'], ds['Vs_theta'])
        ]

    def _setup_dual_ode_mode(self, configurations):
        """Set up dual ODE mode with enhanced config swapping."""
        self.dual_system = DualODESystem(configurations=configurations, rng=self.rng)
        
        # Pre-generate first trajectory
        self._regenerate_current_trajectory()

    def _setup_random_ode_mode(self):
        """Set up legacy random ODE mode."""
        # This maintains compatibility with existing random ODE functionality
        # Implementation would go here if needed
        pass

    def _to_callable(self, data):
        """Convert (t, y) data to callable interpolation function."""
        t, v = data
        t = np.asarray(t)
        v = np.asarray(v)
        interp = interp1d(t, v, kind='linear', fill_value='extrapolate', assume_sorted=True)
        t_min, t_max = (self.t_min, self.t_max)

        def f(x):
            x = np.asarray(x)
            if t_min is not None and (x < t_min).any():
                raise ValueError("Time values below t_min")
            if t_max is not None and (x > t_max).any():
                raise ValueError("Time values above t_max")
            return interp(x)

        return f

    def _regenerate_current_trajectory(self):
        """Regenerate current trajectory for dual ODE mode."""
        if self.mode != "dual_ode":
            return
            
        t_span = (self.t_min, self.t_max)
        
        # Use the enhanced solve method with automatic parameter updates
        t, voltage, theta, traj_info = self.dual_system.solve_with_param_update(
            t_span, 
            self.num_of_points, 
            swap_config=self.auto_config_swap,
            swap_mode=self.swap_mode
        )
        
        # Store current trajectory data
        self.current_t = t
        self.current_voltage = voltage
        self.current_theta = theta
        self.current_traj_info = traj_info
        
        # Create interpolation functions
        self.current_f_Vs = self._to_callable((t, voltage))
        self.current_f_Th = self._to_callable((t, theta))

    def __call__(self, t, randomize=False):
        """Call interface - get voltage and theta values at time t."""
        if self.mode == "dataset":
            idx = self.rng.integers(len(self._dataset_funcs))
            fV, fTh = self._dataset_funcs[idx]
            return fV(t), fTh(t)
        
        elif self.mode == "dual_ode":
            # Only regenerate trajectory if explicitly requested, not during regular ODE solving
            # This prevents infinite recursion during solve_ivp calls
            if randomize and (self.auto_param_update or self.auto_config_swap):
                self._regenerate_current_trajectory()
            
            return self.current_f_Vs(t), self.current_f_Th(t)
        
        else:  # random_ode or other modes
            raise NotImplementedError(f"__call__ not implemented for mode '{self.mode}'")

    def sample(self, num, t_span=None, force_new_trajectory=False):
        """
        Sample trajectory points.
        
        Parameters:
        - num: number of points to sample
        - t_span: time span (optional)
        - force_new_trajectory: if True, force generation of new trajectory
        
        Returns:
        - t: time array
        - Vs: voltage array  
        - Th: theta array
        - info: trajectory information (for dual_ode mode)
        """
        if self.mode == "dataset":
            if t_span is None:
                if self.t_min is None or self.t_max is None:
                    raise ValueError("Must provide t_span or set t_min/t_max")
                t_span = (self.t_min, self.t_max)
            t = np.linspace(*t_span, num)
            Vs, Th = self(t)
            return t, Vs, Th
            
        elif self.mode == "dual_ode":
            # Generate new trajectory if needed
            if force_new_trajectory or (self.auto_param_update and not hasattr(self, 'current_t')):
                self._regenerate_current_trajectory()
            
            if t_span is None:
                t_span = (self.t_min, self.t_max)
            
            t = np.linspace(*t_span, num)
            Vs = self.current_f_Vs(t)
            Th = self.current_f_Th(t)
            
            return t, Vs, Th
            
        else:
            raise NotImplementedError(f"sample not implemented for mode '{self.mode}'")

    def sample_trajectory(self):
        """Sample a complete trajectory (mainly for dataset mode compatibility)."""
        if self.mode == 'dataset':
            idx = self.rng.integers(len(self.dataset['Vs']))
            return {
                't': self.dataset['t'],
                'Vs': self.dataset['Vs'][idx],
                'Vs_theta': self.dataset['Vs_theta'][idx]
            }
        elif self.mode == 'dual_ode':
            # For dual_ode mode, return current trajectory
            return {
                't': self.current_t,
                'Vs': self.current_voltage,
                'Vs_theta': self.current_theta,
                'info': self.current_traj_info
            }

    # ===== DUAL ODE SPECIFIC METHODS =====

    def swap_config(self, config_index=None, mode='sequential'):
        """Swap configuration in dual ODE mode."""
        if self.mode != "dual_ode":
            raise RuntimeError("swap_config only available in 'dual_ode' mode")
        
        old_config = self.dual_system.current_config_index
        new_config, config_name = self.dual_system.swap_config(config_index, mode)
        
        # Regenerate trajectory with new configuration
        self._regenerate_current_trajectory()
        
        return old_config, new_config, config_name

    def force_new_trajectory(self, swap_config=False, swap_mode='sequential'):
        """Force generation of a new trajectory with parameter updates."""
        if self.mode != "dual_ode":
            raise RuntimeError("force_new_trajectory only available in 'dual_ode' mode")
        
        # Temporarily override auto settings to force regeneration
        old_swap = self.auto_config_swap
        old_mode = self.swap_mode
        
        self.auto_config_swap = swap_config
        self.swap_mode = swap_mode
        
        self._regenerate_current_trajectory()
        
        # Restore original settings
        self.auto_config_swap = old_swap
        self.swap_mode = old_mode
        
        return self.current_traj_info

    def get_dual_system_info(self):
        """Get information about the dual ODE system."""
        if self.mode == "dual_ode":
            return self.dual_system.get_info()
        return None

    def get_trajectory_info(self):
        """Get information about current trajectory."""
        if self.mode == "dual_ode" and hasattr(self, 'current_traj_info'):
            return self.current_traj_info
        return None

    @property
    def steady(self):
        """
        Property that returns a function to get uniform steady state values.
        
        This reads the reference values from the configurations and validates that 
        they are consistent across all configurations. If ref_values differ between 
        configs, raises an error.
        
        Usage:
            steady_func = enhanced_ext_input.steady
            voltage_ref, theta_ref = steady_func()  # Returns config ref values
        
        Or directly:
            voltage_ref, theta_ref = enhanced_ext_input.steady()
        
        Returns:
            callable: Function that returns uniform steady state values as tuple (voltage, theta)
        """
        def get_uniform_steady_state():
            """
            Get uniform steady state reference values from configurations.
            
            Validates that all configurations have the same ref_value for voltage and theta.
            If they differ, raises an error. Returns the consistent reference values.
            
            Returns:
                tuple: (voltage_ref, theta_ref) from configurations
                
            Raises:
                RuntimeError: If configurations have different ref_value settings
            """
            if self.mode != "dual_ode":
                raise RuntimeError("steady property only available in 'dual_ode' mode")
            
            if not hasattr(self, 'dual_system') or not hasattr(self.dual_system, 'configurations'):
                raise RuntimeError("dual_system or configurations not initialized")
            
            # Extract reference values from all configurations
            voltage_refs = []
            theta_refs = []
            
            for i, config in enumerate(self.dual_system.configurations):
                # Get voltage ref_value - handle both single value and range
                voltage_ref_config = config['voltage_config']['param_ranges']['ref_value']
                if isinstance(voltage_ref_config, (tuple, list)):
                    if voltage_ref_config[0] != voltage_ref_config[1]:
                        raise RuntimeError(f"Configuration {i} has voltage ref_value range {voltage_ref_config}, but steady state requires fixed reference values")
                    voltage_ref = voltage_ref_config[0]
                else:
                    voltage_ref = voltage_ref_config
                    
                # Get theta ref_value - handle both single value and range  
                theta_ref_config = config['theta_config']['param_ranges']['ref_value']
                if isinstance(theta_ref_config, (tuple, list)):
                    if theta_ref_config[0] != theta_ref_config[1]:
                        raise RuntimeError(f"Configuration {i} has theta ref_value range {theta_ref_config}, but steady state requires fixed reference values")
                    theta_ref = theta_ref_config[0]
                else:
                    theta_ref = theta_ref_config
                    
                voltage_refs.append(voltage_ref)
                theta_refs.append(theta_ref)
            
            # Validate that all configurations have the same reference values
            voltage_ref_uniform = voltage_refs[0]
            theta_ref_uniform = theta_refs[0]
            
            for i, (v_ref, t_ref) in enumerate(zip(voltage_refs, theta_refs)):
                if v_ref != voltage_ref_uniform:
                    raise RuntimeError(f"Configuration {i} has voltage ref_value={v_ref}, but configuration 0 has {voltage_ref_uniform}. All configurations must have the same ref_values for uniform steady state.")
                if t_ref != theta_ref_uniform:
                    raise RuntimeError(f"Configuration {i} has theta ref_value={t_ref}, but configuration 0 has {theta_ref_uniform}. All configurations must have the same ref_values for uniform steady state.")
            
            # Return the validated uniform reference values
            return (float(voltage_ref_uniform), float(theta_ref_uniform))
        
        return get_uniform_steady_state


    def print_status(self):
        """Print current status of the external input system."""
        print(f"=== ExternalInput Status ===")
        print(f"Mode: {self.mode}")
        print(f"Auto Parameter Update: {self.auto_param_update}")
        print(f"Auto Config Swap: {self.auto_config_swap}")
        
        if self.mode == "dual_ode":
            print(f"Swap Mode: {self.swap_mode}")
            self.dual_system.print_info()
            
            if hasattr(self, 'current_traj_info'):
                print(f"\nCurrent Trajectory:")
                print(f"  Number: {self.current_traj_info['trajectory_number']}")
                print(f"  Config: {self.current_traj_info['config_name']}")

    # ===== BACKWARD COMPATIBILITY METHODS =====

    def regenerate_with_new_params(self, voltage_only=False, theta_only=False):
        """Legacy method - regenerate with new parameters."""
        if self.mode == "dual_ode":
            if voltage_only or theta_only:
                # For specific parameter updates, manually reset then regenerate
                if voltage_only:
                    self.dual_system.voltage_ode.reset_params()
                if theta_only:
                    self.dual_system.theta_ode.reset_params()
            self._regenerate_current_trajectory()
        else:
            raise RuntimeError("regenerate_with_new_params only valid in 'dual_ode' mode")


# ========================= RANDOMIZED SYSTEMS =========================

class RandomODESystem:
    """
    Randomly switches between 1st and 2nd order ODEs for both voltage and theta
    when generating new trajectories.
    """

    def __init__(self, voltage_order_prob=0.5, theta_order_prob=0.5,
                 voltage_configs=None, theta_configs=None, rng=None):
        """
        Parameters:
        - voltage_order_prob: probability of using 2nd order for voltage
        - theta_order_prob: probability of using 2nd order for theta
        - voltage_configs/theta_configs: dict with 'first_order'/'second_order'
        - rng: numpy.random.Generator
        """
        self.voltage_order_prob = float(voltage_order_prob)
        self.theta_order_prob = float(theta_order_prob)
        self.rng = rng or np.random.default_rng()

        # Default voltage configurations
        if voltage_configs is None:
            voltage_configs = {
                'first_order': {
                    'param_ranges': {
                        'ref_value': (0.8, 1.2),
                        'k': (1.0, 3.0),
                        'noise_amplitude': (0.0, 0.05),
                        'omega_noise': (0.5, 3.0)
                    },
                    'y0_ranges': {
                        'initial_value': (0.6, 1.0)
                    }
                },
                'second_order': {
                    'param_ranges': {
                        'ref_value': (0.9, 1.1),
                        'zeta': (0.1, 0.5),
                        'omega_n': (5, 15)
                    },
                    'y0_ranges': {
                        'initial_value': (0.8, 1.2),
                        'initial_derivative': (-0.1, 0.1)
                    }
                }
            }

        # Default theta configurations
        if theta_configs is None:
            theta_configs = {
                'first_order': {
                    'param_ranges': {
                        'ref_value': (0.0, 0.0),
                        'k': (0.5, 2.0),
                        'noise_amplitude': (0.0, 0.1),
                        'omega_noise': (0.1, 1.0)
                    },
                    'y0_ranges': {
                        'initial_value': (-0.2, 0.2)
                    }
                },
                'second_order': {
                    'param_ranges': {
                        'ref_value': (0.0, 0.0),
                        'zeta': (0.1, 0.4),
                        'omega_n': (2, 8)
                    },
                    'y0_ranges': {
                        'initial_value': (-0.3, 0.3),
                        'initial_derivative': (-0.5, 0.5)
                    }
                }
            }

        self.voltage_configs = voltage_configs
        self.theta_configs = theta_configs

        # Current system (set when generating)
        self.current_dual_system = None
        self._generate_random_system()

    def _generate_random_system(self):
        """Generate a new random dual ODE system."""
        v_second = self.rng.random() < self.voltage_order_prob
        t_second = self.rng.random() < self.theta_order_prob

        voltage_order = 'second_order' if v_second else 'first_order'
        theta_order = 'second_order' if t_second else 'first_order'

        voltage_config = {'type': voltage_order, **self.voltage_configs[voltage_order]}
        theta_config = {'type': theta_order, **self.theta_configs[theta_order]}

        self.current_dual_system = DualODESystem(voltage_config, theta_config, rng=self.rng)
        return voltage_order, theta_order

    def generate_trajectory(self, t_span, num_points=1000):
        """
        Generate a new trajectory with randomly selected ODE orders.

        Returns:
        - t: time array
        - voltage: voltage solution
        - theta: theta solution
        - info: dict with order information and parameters
        """
        v_order, th_order = self._generate_random_system()
        t, voltage, theta = self.current_dual_system.solve(t_span, num_points)
        info = self.current_dual_system.get_info()
        info['voltage_order'] = v_order
        info['theta_order'] = th_order
        return t, voltage, theta, info

    def get_current_info(self):
        """Get information about current system."""
        if self.current_dual_system:
            info = self.current_dual_system.get_info()
            info['voltage_order'] = info['voltage']['type']
            info['theta_order'] = info['theta']['type']
            return info
        return None

    def print_current_info(self):
        """Print current system configuration."""
        info = self.get_current_info()
        if info:
            print("=== Random ODE System Current Configuration ===")
            print(f"\nVoltage: {info['voltage_order']} order")
            print(f"  Parameters: {info['voltage']['params']}")
            print(f"  Initial conditions: {info['voltage']['y0']}")
            print(f"\nTheta: {info['theta_order']} order")
            print(f"  Parameters: {info['theta']['params']}")
            print(f"  Initial conditions: {info['theta']['y0']}")


class RandomExternalInput(ExternalInput):
    """
    ExternalInput that randomly switches between ODE orders.
    """

    def __init__(self, mode="random_ode", voltage_order_prob=0.5, theta_order_prob=0.5,
                 voltage_configs=None, theta_configs=None, t_min=None, t_max=None,
                 num_of_points=None, rng=None):
        """
        Parameters:
        - mode: should be 'random_ode'
        - voltage_order_prob: probability of 2nd order for voltage
        - theta_order_prob: probability of 2nd order for theta
        - voltage_configs/theta_configs: configurations for both orders
        - rng: numpy.random.Generator
        """
        self.mode = mode
        self.t_min = t_min or 0
        self.t_max = t_max or 5
        self.num_of_points = num_of_points or 1000
        self.rng = rng or np.random.default_rng()

        if mode == "random_ode":
            self.random_system = RandomODESystem(
                voltage_order_prob=voltage_order_prob,
                theta_order_prob=theta_order_prob,
                voltage_configs=voltage_configs,
                theta_configs=theta_configs,
                rng=self.rng
            )
            self._generate_current_trajectory()
        else:
            # Fall back to parent class for other modes
            super().__init__(mode=mode, t_min=t_min, t_max=t_max, num_of_points=num_of_points, rng=self.rng)

    def _generate_current_trajectory(self):
        """Generate current trajectory with random ODE orders."""
        t_span = (self.t_min, self.t_max)
        self.current_t, self.current_voltage, self.current_theta, self.current_info = \
            self.random_system.generate_trajectory(t_span, self.num_of_points)

        # Create interpolation functions
        self.current_f_Vs = self._to_callable((self.current_t, self.current_voltage))
        self.current_f_Th = self._to_callable((self.current_t, self.current_theta))

    def sample(self, num, t_span=None, regenerate=False):
        """
        Sample from current trajectory.

        Parameters:
        - num: number of points
        - t_span: time span (optional)
        - regenerate: if True, generate new random trajectory first
        """
        if self.mode != "random_ode":
            return super().sample(num, t_span)

        if regenerate:
            self._generate_current_trajectory()

        if t_span is None:
            t_span = (self.t_min, self.t_max)

        t = np.linspace(*t_span, num)
        Vs = self.current_f_Vs(t)
        Th = self.current_f_Th(t)
        return t, Vs, Th

    def regenerate_trajectory(self):
        """Generate a new trajectory with random ODE orders."""
        if self.mode == "random_ode":
            self._generate_current_trajectory()
            return self.current_info
        else:
            raise RuntimeError("regenerate_trajectory only valid in 'random_ode' mode")

    def get_current_trajectory_info(self):
        """Get information about current trajectory."""
        if self.mode == "random_ode":
            return self.current_info
        return None

    def print_current_trajectory_info(self):
        """Print current trajectory information."""
        info = self.get_current_trajectory_info()
        if info:
            print("=== Current Random Trajectory Info ===")
            print(f"\nVoltage: {info['voltage_order']} order")
            print(f"  Parameters: {info['voltage']['params']}")
            print(f"  Initial conditions: {info['voltage']['y0']}")
            print(f"\nTheta: {info['theta_order']} order")
            print(f"  Parameters: {info['theta']['params']}")
            print(f"  Initial conditions: {info['theta']['y0']}")

