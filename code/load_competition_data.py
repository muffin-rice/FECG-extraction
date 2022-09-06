import base64
import os
from copy import copy
from os.path import isfile

import numpy as np
import scipy
import scipy.io
import torch
from pytorch_lightning import LightningDataModule
from scipy.signal import decimate
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from denoising import wav_filt
from utils import return_peaks, gauss_kernel
import wfdb

DROP_LAST = False
LOAD_INTO_MEMORY = False

PEAK_WINDOW = 5
PEAK_SIGMA = 1

class RealECGDataset(Dataset):
    def __init__(self, indices):
        super(RealECGDataset, self).__init__()
        signals = np.array(
            tuple(
                map(
                    lambda x: x.p_signal[:, 0],
                    (wfdb.io.rdrecord(f'/Users/Richard/Documents/fECG_research/ResearchCode/dataset-a/a{i:02}')
                    for i in indices)
                )
            )
        )
        signals = np.reshape(signals, (signals.shape[0] * (signals.shape[1] // 4000), 4000))
        filt_signals, _, _, _ = wav_filt(signals, lvl=4, winlen=41, polyorder=4)
        filt_signals = filt_signals[0]
        sig_reg = filt_signals - filt_signals.mean(axis=-1, keepdims=True)
        sig_reg /= np.abs(sig_reg).max(axis=-1, keepdims=True)
        sig_reg = decimate(sig_reg, 8, axis=-1)
        self.signals = sig_reg

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            idx = [idx]

        inp = self.signals[idx]
        return inp


class RealECGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, window_size, dataset_type: str, batch_size: int,
                 num_workers: int , pin_memory: bool = False):
        super().__init__()
        idx = [i for i in range(1, 76) if i not in {33, 38, 47, 52, 54, 71, 74}]
        self.train_idx, test_valid_idx = train_test_split(idx, test_size=0.2, train_size=0.8, random_state=1)
        test_len = len(test_valid_idx) // 2
        self.test_idx = test_valid_idx[:test_len]
        self.val_idx = test_valid_idx[test_len:]
        self.window_size = window_size
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        data = RealECGDataset(
            self.train_idx
        )

        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=True,
            drop_last=DROP_LAST
        )

    def val_dataloader(self):
        data = RealECGDataset(
            self.val_idx
        )

        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
            drop_last=DROP_LAST
        )

    def test_dataloader(self):
        data = RealECGDataset(
            self.test_idx
        )

        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
            drop_last=True
        )


