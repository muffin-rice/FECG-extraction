import os
from copy import copy
from os.path import isfile

import numpy as np
import scipy
import scipy.io
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


DROP_LAST = False
LOAD_INTO_MEMORY = False


def normalize(batch, batch_extra):  # normalizes the signal

    scale_factor = np.repeat(np.ptp(batch, axis=-1, keepdims=True), batch.shape[-1], axis=-1)
    shift = np.repeat(np.min(batch, axis=-1, keepdims=True), batch.shape[-1], axis=-1)
    batch = (batch - shift) / scale_factor
    batch_extra = (batch_extra - shift) / scale_factor

    return batch, batch_extra


def scale_signals(mecg_sig, fecg_sig):
    mecg, fecg = copy(mecg_sig), copy(fecg_sig)
    aecg = mecg + fecg
    aecg, mecg = normalize(aecg, mecg)
    return mecg, aecg - mecg


def stft(sig: np.array):
    if len(sig.shape) == 3:
        x = []
        for isig in range(sig.shape[0]):  # iterate through batch
            stft_image = torch.stft(torch.from_numpy(sig[isig, ...]), n_fft=32, normalized=True, hop_length=1,
                                    onesided=True, return_complex=True, center=False)

            x.append(torch.cat((stft_image.real, stft_image.imag), dim=1).unsqueeze(0))

        return torch.cat(x)

    else:
        stft_image = torch.stft(torch.from_numpy(sig), n_fft=32, normalized=True, hop_length=1,
                                onesided=True, return_complex=True, center=False)

        return torch.cat((stft_image.real, stft_image.imag), dim=1)


def load_data(data_dir: str):
    dataset = []
    fnames = os.listdir(data_dir)
    for fname in fnames:
        true_fname = f'{data_dir}/{fname}'
        if isfile(true_fname) and fname != '.DS_Store':
            dataset.append(true_fname)

    # train_data, test_val_data = train_test_split(dataset, test_size=0.5)
    # test_data, val_data = train_test_split(test_val_data, test_size=0.5)

    return np.array(dataset, dtype=object)


from scipy.signal import savgol_filter, filtfilt, firwin, firwin2


def filt(signal: np.ndarray, numtaps=31, sampling_rate=125):
    '''removes the baseline + power line noise'''

    Fs = sampling_rate

    gain = [0, 1, 1, 0, 0]
    freq = [0, 1, 45, 55, Fs / 2]

    for i in range(signal.shape[0]):
        b = firwin2(numtaps, freq, gain, fs=Fs, window='hamming', antisymmetric=True)
        window = filtfilt(b, 1, signal[i, :])
        signal[i, :] = window

    return signal


def list_files(directory: str):
    dataset = list(filter(
        os.path.isfile,
        map(
            lambda f: os.path.join(directory, f),
            filter(
                lambda f: f != '.DS_Store',
                os.listdir(directory)
            )
        )
    ))

    return np.array(dataset)  # defaults to Unicode string of minimum length


class ECGDataset(Dataset):
    def __init__(self, window_size, data_dir, dataset_type: str = None, split: str = 'train',
                 load_into_memory=LOAD_INTO_MEMORY):
        super(ECGDataset, self).__init__()
        # self.train_data, self.val_data, self.test_data = load_data(data_dir)
        self.dataset = load_data(f'{data_dir}/{split}')

        self.window_size = window_size  # window size

        self.load_into_memory = False
        if load_into_memory:
            self.loaded_dataset = []
            for data_i in range(len(self.dataset)):
                self.loaded_dataset.append(self[data_i])

            self.load_into_memory = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.load_into_memory:
            return self.loaded_dataset[idx]

        inp = scipy.io.loadmat(self.dataset[idx])

        keys_to_remove = []
        for k, v in inp.items():
            if type(v) != np.ndarray or 'fname' in k:
                keys_to_remove.append(k)
            else:
                inp[k] = v.astype('double')

        for key in keys_to_remove:
            inp.pop(key)

        inp['mecg_sig'], inp['fecg_sig'] = scale_signals(inp['mecg_sig'], inp['fecg_sig'])

        inp['mecg_stft'] = stft(inp['mecg_sig'])
        inp['fecg_stft'] = stft(inp['fecg_sig'])

        # assert inp['mecg_stft'].shape[0] == 2

        if torch.isnan(inp['mecg_stft']).any() or torch.isnan(inp['mecg_stft']).any():
            print(inp)
            raise SystemError

        return inp


class ECGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, window_size, dataset_type: str, batch_size: int,
                 num_workers: int , pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        data = ECGDataset(
            data_dir=self.data_dir,
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
            persistent_workers=True,
            drop_last=DROP_LAST
        )

    def val_dataloader(self):
        data = ECGDataset(
            data_dir=self.data_dir,
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
            persistent_workers=True,
            drop_last=DROP_LAST
        )

    def test_dataloader(self):
        data = ECGDataset(
            data_dir=self.data_dir,
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
            persistent_workers=True,
            drop_last=True
        )


