from functions import *
import torch
import wandb
import time
from src.dataset.create_dataset_functions import ODE_modelling, gen_dataset
from src.dataset.sm_models_o_e import SynchronousMachineModels

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from omegaconf import OmegaConf
from src.dataset.ext_func6 import *

# --- Initialize wandb ---
wandb.init(
    project="Physics-Informed-Neural-Operator",
    name="dataset_generation",
    config={
        "config_1": "src/conf/ext_func/config_1.yaml",
        "config_2": "src/conf/ext_func/config_2.yaml",
        "setup_dataset": "src/conf/setup_dataset_o_e.yaml"
    }
)

# --- Load configs ---
config_1 = OmegaConf.load("src/conf/ext_func/config_1.yaml")
config_2 = OmegaConf.load("src/conf/ext_func/config_2.yaml")
cfg = OmegaConf.load("src/conf/setup_dataset_o_e.yaml")

wandb.config.update(OmegaConf.to_container(cfg, resolve=True))  # log cfg parameters

set_torch_precision(use_float64=cfg.flag_for_precision)  # Set precision for torch tensors

# --- Create enhanced ExternalInput ---
enhanced_ext_input = ExternalInput(
    mode="dual_ode",
    configurations=[config_1, config_2],
    t_min=0,
    t_max=cfg.time,
    num_of_points=cfg.num_of_points,
    auto_param_update=True,
    auto_config_swap=True,
    swap_mode='random'
)

# --- Model and dataset setup ---
SM_model = ODE_modelling(cfg)
modelling_full = SynchronousMachineModels(cfg)

datasets = ["train_labelled", "train_colloc", "val_labelled", "test_labelled"]

for i in datasets:
    start_time = time.time()
    if 'labelled' in i:
        flag_create_dataset = 'labelled'
    elif 'colloc' in i:
        flag_create_dataset = 'collocation'
    
    limit = cfg.dataset[i + '_traj']
    print(f"Creating {i} dataset with {limit} trajectories...")
    
    data = gen_dataset(
        SM_model,
        modelling_full,
        enhanced_ext_input,
        flag_create_dataset=flag_create_dataset,
        limit=limit,
        dataset_name=i
    )

    elapsed = time.time() - start_time
    print(f"{i} dataset created in {elapsed:.2f} seconds.")
    
    # Log dataset info to wandb
    wandb.log({
        "dataset": i,
        "trajectories": limit,
        "creation_time_sec": elapsed
    })

wandb.finish()
