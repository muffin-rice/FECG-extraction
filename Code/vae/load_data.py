from torch.utils.data import DataLoader, Dataset
import scipy.io
import scipy
import scipy.signal as ss
import numpy as np
import torch
import os
from os.path import isfile
import wfdb
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

PAIRED_DATA_DIR = 'Data/preprocessed_data/paired_data'
NUM_WORKERS = 8
DROP_LAST = False


def load_data(data_dir: str):
    dataset = []
    fnames = os.listdir(data_dir)
    for fname in fnames:
        true_fname = f'{data_dir}/{fname}'
        if isfile(true_fname) and fname != '.DS_Store':
            dataset.append(true_fname)

    #train_data, test_val_data = train_test_split(dataset, test_size=0.5)
    #test_data, val_data = train_test_split(test_val_data, test_size=0.5)

    return np.array(dataset, dtype = object)


class ECGDataset(Dataset):
    def __init__(self, window_size, data_dir = PAIRED_DATA_DIR, dataset_type: str = None, split: str = 'train'):
        super(ECGDataset, self).__init__()
        # self.train_data, self.val_data, self.test_data = load_data(data_dir)
        self.dataset = load_data(f'{data_dir}/{split}')

        self.window_size = window_size  # window size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inp = scipy.io.loadmat(self.dataset[idx])
        # keys_to_remove = ['__header__', '__version__', '__globals__']
        # for key in keys_to_remove:
        #     inp.pop(key)
        keys_to_remove = []
        for k, v in inp.items():
            if type(v) != np.ndarray or 'fname' in k:
                keys_to_remove.append(k)
            else:
                inp[k] = v.astype('double')

        for key in keys_to_remove:
            inp.pop(key)

        aecg_stft, fecg_stft = inp['aecg_stft'], inp['fecg_stft']

        assert aecg_stft.shape[0] == 2

        if np.isnan(aecg_stft).any() or np.isnan(fecg_stft).any():
            print(inp)
            quit()

        return inp


class ECGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, window_size, dataset_type: str = None, batch_size: int = 8,
                 num_workers: int = NUM_WORKERS, pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        data = ECGDataset(
            window_size=self.window_size,
            dataset_type=self.dataset_type,
            split='train'
        )

        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=DROP_LAST
        )

    def val_dataloader(self):
        data = ECGDataset(
            window_size=self.window_size,
            dataset_type=self.dataset_type,
            split='validation'
        )

        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=DROP_LAST
        )

    def test_dataloader(self):
        data = ECGDataset(
            window_size=self.window_size,
            dataset_type=self.dataset_type,
            split='test'
        )

        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True
        )
