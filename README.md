# PowerDeepONet

Physics-informed neural operators for power-system models. This repository contains tools to generate ODE datasets for synchronous machine models and to train DeepONet / PINN-style neural operators used in the accompanying paper.

## Reproduce results (quick start)

These steps reproduce the core experiments used in the paper. They assume you have Python 3.8+ and the repository cloned.

1) Install dependencies

```
pip install -r requirements.txt
```

2) Generate datasets

Run the dataset generation script. By default this uses the YAML config files under `src/conf/`.

```
python datasets.py
```

If you need to change dataset settings (time span, model, sampling, output paths) edit the configuration files in `src/conf/` before running. Common configs:
- `src/conf/setup_dataset_o_e.yaml`
- `src/conf/setup_dataset_nn_o_e.yaml`

3) Train and test models

After the datasets are prepared, run the training/test script used for the paper:

```
python final_train_test.py
```

This script reads the neural-network configuration and dataset paths from the YAML files in `src/conf/` (see `setup_nn_model.yaml` and the `dataset/` configs). If you use Weights & Biases for experiment tracking, ensure `wandb` is configured or disable it in the config.

## Configuration

- All experiment settings are YAML-based and located in `src/conf/`.
- Edit `setup_nn_model.yaml` to change network architecture, optimizer, or training hyperparameters.
- Edit the dataset setup files (`setup_dataset_*`) to change sampling, time discretization, and model selection.

## Files of interest

- `datasets.py` — high-level script to generate datasets used in experiments.
- `final_train_test.py` — runs training and evaluation used for the paper.
- `src/dataset/` — dataset generation helpers and power-system ODE models.
- `src/neural_operator/` — neural operator models, training loops, and evaluation utilities.

## Tips

- Use a virtual environment (venv or conda) to isolate dependencies.
- For reproducible results, pin package versions in `requirements.txt` (I can help propose pins).
- Long-running dataset generation and training can be configured to use a GPU; ensure PyTorch is installed with CUDA support if available.
- To reduce terminal noise for a paper release, set logging level or edit the verbose flags in the YAML configs.

## Citation

If you use this code in a publication, please cite:

Neural Operators for Power Systems: A Physics-Informed Framework for Modeling Power System Components

Ioannis Karampinis, Petros Ellinas, Johanna Vorwerk and Spyros Chatzivasileiadis

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

