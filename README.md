# MGPVAE

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