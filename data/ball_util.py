import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import os
from os import path
from glob import glob

class SequenceDataset(data.Dataset):
    def __init__(
        self,
        sequence_dir,
    ):
        self.sequence_dir = sequence_dir
        self.sequence_labels = self.sequence_dir
        try:
            self.sequence_filenames = glob(f"{self.sequence_dir}/*")
        except:
            raise ValueError("dir incorrect")

    def __len__(self):
        return len(self.sequence_filenames)

    def __getitem__(self, idx):
        sequence_path = self.sequence_filenames[idx]

        stored_obj = np.load(sequence_path)
        sequence = stored_obj["sequence"]
        missing_sequence = stored_obj["missing_sequence"]
        t = stored_obj["t"]
        missing_mask = stored_obj["missing_mask"]
        seq_length_train = stored_obj["seq_length"]
        return missing_sequence.astype(np.float64), t, missing_mask.astype(np.float64), seq_length_train, sequence.astype(np.float64)

    def plot(self, idx=0, figsize=(20, 30), show=True):
        img_path = self.img_filenames[idx]
        img = np.load(img_path)["images"]
        fig = plt.figure(figsize=figsize)
        if show:
            plt.imshow(np.concatenate(img, axis=1))
        return fig


def numpy_collate(batch):
    """From https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html"""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyDataset(data.Dataset):
    def __init__(
        self, img_dir, train_foldername="train", mask_foldername="mask", transform=None
    ):
        self.img_dir = path.join(img_dir, "groundtruth")
        self.img_labels = self.img_dir
        self.train_dir = path.join(img_dir, train_foldername)
        self.mask_dir = path.join(img_dir, mask_foldername)
        try:
            self.img_filenames = glob(f"{self.img_dir}/*")
        except:
            raise ValueError("img_dir incorrect")
        try:
            self.train_filenames = glob(f"{self.train_dir}/*")
        except:
            raise ValueError("train_dir incorrect")
        try:
            self.mask_filenames = glob(f"{self.mask_dir}/*")
        except:
            raise ValueError("mask_dir incorrect")

        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = self.img_filenames[idx]
        train_path = self.train_filenames[idx]
        mask_path = self.mask_filenames[idx]

        y_train = np.load(train_path)
        mask_train = np.load(mask_path)
        y_test = np.load(img_path)
        if self.transform:
            y_train = self.transform(y_train)
            y_test = self.transform(y_test)
        return y_train, mask_train, y_test


class NumpyLoader(data.DataLoader):
    """From https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html"""

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
