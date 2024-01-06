import sys 
sys.path.append("..")
import os
import numpy as np
from latent_ode.lib import utils
from latent_ode.mujoco_physics import HopperPhysics
import torch
from os import path 
from tqdm import tqdm

data_path = "data/mujoco_1000_slower"
os.makedirs(path.join(data_path, "missing_sequences_1000_slower"), exist_ok=True)
mode = "missing"
missing_prob = 0.4
extrap = False
timepoints = 1000

dataset_obj = HopperPhysics(root='../data/mujoco_1000_slower', download=False, generate=True)
dataset = dataset_obj.get_dataset()
dataset = dataset.numpy()
n_tp_data = dataset[:].shape[1]
## only take first 100 time points for computational ease
dataset = dataset[:, :timepoints, :]

if mode == "missing":
    # Creating dataset for interpolation
    # sample time points from different parts of the timeline, 
    # so that the model learns from different parts of hopper trajectory
    n_traj = len(dataset)
    n_tp_data = dataset.shape[1]
    n_reduced_tp = timepoints

    # sample time points from different parts of the timeline, 
    # so that the model learns from different parts of hopper trajectory
    start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
    end_ind = start_ind + n_reduced_tp

    for i in tqdm(range(dataset.shape[0])):

        ## for MGPVAE
        sequence = dataset[i].copy()
        missing_frames = np.random.binomial(1, missing_prob, size=(timepoints-2))
        missing_frames = np.array([1] + list(missing_frames) + [1]) 
        sequence = sequence[missing_frames==1]
        t_input = np.linspace(0, timepoints-1, timepoints)[:, None][missing_frames==1]
        seq_length = t_input.shape[0]
        sequence = np.concatenate([sequence, np.zeros((timepoints-seq_length, sequence.shape[-1]))], axis=0)
        t = np.ones((timepoints,1))*(timepoints - 1)
        t[:seq_length, :] = t_input
        missing_mask = np.ones(sequence.shape)
        missing_mask[seq_length:] = 0
        np.savez(path.join(data_path, "missing_sequences_1000_slower", f"{i}"), sequence=dataset[i], missing_sequence=sequence.astype(np.float64), t=t, missing_mask=missing_mask, seq_length=seq_length, label=f"{i}")