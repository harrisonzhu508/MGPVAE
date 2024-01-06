# MGPVAE
This repository contains the code for the paper [Markovian Gaussian Process Variational Autoencoders](https://arxiv.org/pdf/2207.05543.pdf).

The implementation is derived and modified from [BayesNewton](https://github.com/AaltoML/BayesNewton/tree/main).

## Requirements
```
matplotlib
numpy
tqdm
ipywidgets
ipython 
ipykernel
torch
torchvision
scikit-learn
dm_control
pandas
```

To install JAX `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

Then `pip install objax`.

## Minimal working examples

Run `python create_mujoco+missing_frame.py` to create some MGPVAE training/val/test data.

Notebooks containing training and inference scripts
- `example_mujoco.ipynb`
- `example_spatiotemporal.ipynb`