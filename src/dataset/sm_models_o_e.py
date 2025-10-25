import numpy as np
from scipy.integrate import solve_ivp
from functions import set_time
import torch
import os
from omegaconf import OmegaConf

class SynchronousMachineModels():
    def __init__(self, config):
        """
        Initialize the synchronous machine model with Automatic Voltage Regulator (AVR).

        Parameters:
            t (float): The current time.
            num_of_points (int): The number of points to evaluate the solution at.
            machine_params (list): The parameters of the synchronous machine.
            system_params (object): The system parameters of the synchronous machine.
            model_flag (str): The flag indicating the type of model.

        Attributes:
            t (float): Time for solving ode.
            machine_params (list): The parameters of the synchronous machine.
            system_params (object): The system parameters of the synchronous machine.
            model_flag (str): The flag indicating the type of model.
            model (str): The type of synchronous machine model.

            X_d_dash (float): The synchronous reactance in the direct axis.
            X_q_dash (float): The synchronous reactance in the quadrature axis.
            Vs (float): The stator voltage magnitude.
            theta_vs (float): The stator voltage angle.
            omega_B (float): The base angular frequency.
            H (float): The inertia constant.
            P_m (float): The mechanical power input (only if governor exists).
            D (float): The damping coefficient.
            T_q_dash (float): The transient reactance time constant (not for infinite bus model).
            X_q (float): The synchronous reactance in the quadrature axis (not for infinite bus model).
            T_d_dash (float): The transient reactance time constant (not for infinite bus model).
            X_d (float): The synchronous reactance in the direct axis (not for infinite bus model).
            R_s (float): The stator resistance (not for infinite bus model).
            E_fd (float): The field voltage (only if AVR exists).
            K_A (float): The AVR gain (only if AVR exists).
            T_A (float): The AVR time constant (only if AVR exists).
            K_E (float): The AVR excitation system gain (only if AVR exists).
            T_E (float): The AVR excitation system time constant (only if AVR exists).
            K_F (float): The AVR feedback gain (only if AVR exists).
            T_F (float): The AVR feedback time constant (only if AVR exists).
            V_ref (float): The reference voltage (only if AVR exists).
            P_c (float): The governor control signal (only if governor exists).
            R_d (float): The governor droop coefficient (only if governor exists).
            T_ch (float): The governor time constant (only if governor exists).
            T_sv (float): The governor servo time constant (only if governor exists).
            

        """
        self.time_horizon = config.time
        self.num_of_points = config.num_of_points
        self.params_dir = config.dirs.params_dir
        self.machine_num = config.model.machine_num
        self.model_flag = config.model.model_flag
        # parameters of the synchronous machine
        self.define_machine_params()
        # parameters of the power system
        self.define_system_params()
          

    def define_machine_params(self):
        """
        Define the parameters of the synchronous machine based on the machine_num
        and potentially the parameters of the AVR and the Governor.

        Returns:
            Tuple: A tuple containing the machine parameters, AVR parameters, and Governor parameters.
                - machine_params (dict): The parameters of the synchronous machine.
                - avr_params (dict): The parameters of the automatic voltage regulator (AVR).
                - gov_params (dict): The parameters of the governor.
        """
        machine_params_path = os.path.join(self.params_dir, "machine" + str(self.machine_num) + ".yaml") # path to the selected machine parameters
        machine_params = OmegaConf.load(machine_params_path)
        if not(self.model_flag=="SM_IB"):
            for param in ['X_d_dash', 'X_q_dash', 'H', 'D', 'T_d_dash', 'X_d', 'T_q_dash', 'X_q', 'E_fd', 'P_m']:
                setattr(self, param, getattr(machine_params, param))
        else:
            for param in ['H', 'D', 'X_d_dash', 'X_q_dash']:
                setattr(self, param, getattr(machine_params, param))

        
        #if not self.model_flag=="SM_AVR_GOV": # if governor exists, then P_m is a control input
        #        self.P_m = machine_params.P_m

        if self.model_flag=="SM6":
            self.model_name = "6th order Synchronous Machine Model"
            for param in ['T_d_dash_dash', 'T_q_dash_dash', 'X_d_dash_dash', 'X_q_dash_dash']:
                setattr(self, param, getattr(machine_params, param))

        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV": 
            avr_params_path= os.path.join(self.params_dir,"avr.yaml") #path to the automatic voltage regulator parameters
            avr_params= OmegaConf.load(avr_params_path)
            for param in ['K_A', 'T_A', 'K_E', 'T_E', 'K_F', 'T_F', 'V_ref']:
                setattr(self, param, getattr(avr_params, param))
        else: 
            avr_params=None

        if self.model_flag=="SM_AVR_GOV":
            gov_params_path= os.path.join(self.params_dir,"gov.yaml") #path to the governor parameters
            gov_params= OmegaConf.load(gov_params_path)
            for param in ['P_c', 'R_d', 'T_ch', 'T_sv']:
                setattr(self, param, getattr(gov_params, param))
        else:
            gov_params=None

        return 
    
    def define_system_params(self):
        """
        Define the parameters of the power system.

        Returns:
            dict: The parameters of the power system.
        """
        system_params_path = os.path.join(self.params_dir, "system_bus.yaml")
        system_params = OmegaConf.load(system_params_path)
        for param in ['omega_B', 'Rs','Re', 'Xep']:
            setattr(self, param, getattr(system_params, param))
        return 

    def calculate_currents(self, theta, E_d_dash, E_q_dash, Vs, theta_vs):
        """
        Calculates the currents I_d and I_q based on the given parameters.

        Parameters:
        theta (rad): The angle .
        E_d_dash (pu): The value of E_d_dash.
        E_q_dash (pu): The value of E_q_dash.

        Returns:
        tuple: A tuple containing the calculated values of I_d and I_q.
        """

        alpha = [[(self.Rs+self.Re), -(self.X_q_dash+self.Xep)], [(self.X_d_dash+self.Xep), (self.Rs+self.Re)]]
        #beta = [[E_d_dash - self.Vs*np.sin(theta-self.theta_vs)], [E_q_dash - self.Vs*np.cos(theta-self.theta_vs)]]
        
        inv_alpha = np.linalg.inv(alpha)
        # Calculate I_d and I_q
        if isinstance(theta, torch.Tensor):
            beta = [[E_d_dash - Vs*torch.sin(theta-theta_vs)], [E_q_dash - Vs*torch.cos(theta-theta_vs)]]
        else:
            beta = [[E_d_dash - Vs*np.sin(theta-theta_vs)], [E_q_dash - Vs*np.cos(theta-theta_vs)]]
            
        I_d= inv_alpha[0][0]*beta[0][0] + inv_alpha[0][1]*beta[1][0]
        I_q= inv_alpha[1][0]*beta[0][0] + inv_alpha[1][1]*beta[1][0]
        #if isinstance(theta, torch.Tensor):
        #    I_d= inv_alpha[0][0]*(E_d_dash - self.Vs*torch.sin(theta-self.theta_vs)) + inv_alpha[0][1]*(E_q_dash - self.Vs*torch.cos(theta-self.theta_vs))
        #    I_q= inv_alpha[1][0]*(E_d_dash - self.Vs*torch.sin(theta-self.theta_vs)) + inv_alpha[1][1]*(E_q_dash - self.Vs*torch.cos(theta-self.theta_vs))
        #else:
        #    I_d= inv_alpha[0][0]*(E_d_dash - self.Vs*np.sin(theta-self.theta_vs)) + inv_alpha[0][1]*(E_q_dash - self.Vs*np.cos(theta-self.theta_vs))
        #    I_q= inv_alpha[1][0]*(E_d_dash - self.Vs*np.sin(theta-self.theta_vs)) + inv_alpha[1][1]*(E_q_dash - self.Vs*np.cos(theta-self.theta_vs))

        #I_t = np.matmul(inv_alpha, beta)
        #I_d = I_t[0][0]
        #I_q = I_t[1][0]
        
        return I_d, I_q

    def calculate_voltages(self, theta, I_d, I_q, Vs, theta_vs):
        """
        Calculate the voltage V_t based on the given inputs, for AVR model

        Parameters:
        theta (rad): The angle in radians.
        I_d (pu): The d-axis current.
        I_q (pu): The q-axis current.

        Returns:
        float: The magnitude of the total voltage V_t(pu).
        """

        if isinstance(theta, torch.Tensor):
            V_d = self.Re * I_d - self.Xep * I_q + Vs * torch.sin(theta - theta_vs)
            V_q = self.Re * I_q + self.Xep * I_d + Vs * torch.cos(theta - theta_vs)
            V_t = torch.sqrt(V_d ** 2 + V_q ** 2)
        else:
            V_d = self.Re * I_d - self.Xep * I_q + Vs * np.sin(theta - theta_vs)
            V_q = self.Re * I_q + self.Xep * I_d + Vs * np.cos(theta - theta_vs)
            V_t = np.sqrt(V_d ** 2 + V_q ** 2)  # equal to Vs
        return V_t
        
    def odequations(self, t, x, Vs_function=None):
        """
        Calculates the derivatives of the state variables for the synchronous machine model.

        Parameters:
            t (float): The current time.
            x (list): A list of state variables, different for each model type.

        Returns:
            list: A list of derivatives, different for each model type.
        """
        if self.model_flag == "SM_IB" or self.model_flag == "SM4":
            theta, omega, E_d_dash, E_q_dash = x
        elif self.model_flag == "SM4_P":
            theta, omega, E_d_dash, E_q_dash, P_m = x
        elif self.model_flag == "SM6":
            theta, omega, E_d_dash, E_q_dash, E_d_dash_dash, E_q_dash_dash = x
        elif self.model_flag == "SM_AVR":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd = x
        elif self.model_flag == "SM_AVR_GOV":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv = x


        # Calculate Vs and theta_vs
        if Vs_function is not None:
            if callable(Vs_function):
               Vs,theta_vs = Vs_function(t)
            else: 
               Vs,theta_vs = Vs_function 
            #Vs,theta_vs = Vs_function(t)
        else:
            raise ValueError("Vs_function must be provided for this model.")
        # Calculate currents from algebraic equations
        I_d, I_q = self.calculate_currents(theta, E_d_dash, E_q_dash, Vs, theta_vs)

        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"): # calculate V_t from algebraic equations
            V_t = self.calculate_voltages(theta, I_d, I_q, Vs, theta_vs)
        
        # Calculate theta derivative
        dtheta_dt = omega

        # Calculate omega derivative
        if self.model_flag=="SM_AVR_GOV" or self.model_flag=="SM4_P": # calculate omega derivative from algebraic equations
            domega_dt = (self.omega_B / (2 * self.H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega)
        else:
            domega_dt = (self.omega_B / (2 * self.H)) * (self.P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega)
        
        # Calculate E_dash derivatives
        if self.model_flag=="SM_IB":
            dE_q_dash_dt = 0
            dE_d_dash_dt = 0
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]
        
        if self.model_flag=="SM4": 
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]
        
        if self.model_flag=="SM4_P": 
            
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dP_m_dt = 0
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dP_m_dt]
        
        if self.model_flag == "SM6":
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dE_q_dash_dash_dt = (1 / self.T_d_dash_dash) * (E_q_dash - E_q_dash_dash + I_d * (self.X_d_dash - self.X_d_dash_dash))
            dE_d_dash_dash_dt = (1 / self.T_q_dash_dash) * (E_d_dash - E_d_dash_dash - I_q * (self.X_q_dash - self.X_q_dash_dash))
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dE_d_dash_dash_dt, dE_q_dash_dash_dt]
        
        # Automatic Voltage Regulator (AVR) dynamics 4.46-4.48
        # Exciter and AVR equations
        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"):
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dR_F_dt      = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
            dV_r_dt      = (1 / self.T_A) * (-V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t))
            dE_fd_dt     = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e**(E_fd*0.55)) * E_fd + V_r)
            
            if self.model_flag=="SM_AVR_GOV":        # Governor equations # recheck it after the meeting
                dP_m_dt  = (1 / self.T_ch) * (-P_m  + P_sv) # 4.110  from dynamics dT_m_dt = - P_m / (2 * H) + P_sv 4.100 + check draw.io
                dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1/self.R_d) *(omega/self.omega_B))
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_m_dt, dP_sv_dt]#4.116 recheck it after the meeting
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt]


    

