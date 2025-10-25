import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import wandb
from src.neural_operator.training_actions import DeepONetActions
from src.dataset.sm_models_o_e import SynchronousMachineModels
from omegaconf import OmegaConf

# Modify PATH environment variable
os.environ["PATH"] = "/sbin:/usr/sbin:" + os.environ.get("PATH","")




def train(config=None):
    run = wandb.init(config=config)
    config = run.config

    # Load base configuration from YAML
    cfg = OmegaConf.load("src/conf/setup_dataset_nn_o_e.yaml")
    cfg.seed = config.seed
    cfg.nn.weighting.weights = config.loss_weights
    cfg.nn.type = config.nn_type
    cfg.nn.optimizer = config.optimizer


    modelling_full = SynchronousMachineModels(cfg)
    r = DeepONetActions(cfg, modelling_full)

    # Train and test
    r.pinn_train('train_labelled_dataset_'+str(cfg.dataset.number)+'.h5', 'train_colloc_dataset_'+str(cfg.dataset.number)+'.h5', 'train_colloc_initial_conditions_dataset_'+str(cfg.dataset.number)+'.h5', 'val_labelled_dataset_'+str(cfg.dataset.number)+'.h5', wandb_run=run)
    r.test_model('test_labelled_dataset_'+str(cfg.dataset.number)+'.h5', flag_time=True, wandb=run)

    run.finish()

if __name__ == "__main__":
    # Define sweep configuration
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "Test_loss",
            "goal": "minimize"
        },
        "parameters": {
            "seed": {"values": [1, 3, 7]},
            "nn_type": {"values": ['DeepONetUnstackedf', 'DeepONetStackedf', 'SinglePINN']},#'DeepONetUnstackedf', 'SinglePINNNet', 'DeepONetStackedf']},
            "loss_weights": {"values": [[1, 1e-3, 1e-4, 1e-3]]},
            "optimizer": {"values": ['SOAP']}
        }
    }

    # Initialize and run sweep
    sweep_id = wandb.sweep(sweep_config, project="Neural-Operator-SM")
    wandb.agent(sweep_id, function=train)