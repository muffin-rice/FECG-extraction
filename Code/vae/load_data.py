from torch.utils.data import DataLoader, Dataset
import scipy.io
import scipy
import numpy as np
import torch
import os
from os.path import isfile
from pytorch_lightning import LightningDataModule
from hyperparams import *
from utils import scale_signals, stft_batch, invert_stft_batch, filt
from utils import return_peaks, gauss_kernel


def load_data(data_dir: str) -> np.array:
    dataset = []
    fnames = os.listdir(data_dir)
    for fname in fnames:
        true_fname = f'{data_dir}/{fname}'
        if isfile(true_fname) and fname != '.DS_Store':
            dataset.append(true_fname)

    # train_data, test_val_data = train_test_split(dataset, test_size=0.5)
    # test_data, val_data = train_test_split(test_val_data, test_size=0.5)

    return np.array(dataset, dtype=object)

def calc_peak_mask(sig : np.array, peak_window = PEAK_WINDOW, peak_sigma = PEAK_SIGMA):
    # signal should be single channel; 1 x sig_legnth
    peaks = return_peaks(sig[:])
    peak_mask = np.zeros(sig.shape)

    for peak in peaks:
        peak_mask[peak-int(peak_window/2) : peak+int(peak_window/2)+1] = gauss_kernel(peak_window, peak_sigma)

    return peak_mask

def calc_multi_channel_peak_mask(sig : np.ndarray, peak_window = PEAK_WINDOW, peak_sigma = PEAK_SIGMA):
    mask = np.zeros(sig.shape)
    for channel in range(sig.shape[0]):
        mask[channel, :] = calc_peak_mask(sig[channel, :], peak_window, peak_sigma)

    return mask


class ECGDataset(Dataset):
    def __init__(self, window_size, data_dir=DATA_DIR, dataset_type: str = None, split: str = 'train',
                 load_into_memory=LOAD_INTO_MEMORY, load_type : str = 'normal'):
        super(ECGDataset, self).__init__()
        # self.train_data, self.val_data, self.test_data = load_data(data_dir)
        self.data_dir = f'{data_dir}/{split}'
        self.dataset = load_data(f'{data_dir}/{split}')

        self.window_size = window_size  # window size

        num_mecg, num_fecg = self.num_mecg_fecg(self.dataset)
        # index starts at 1
        self.mecg_index_list, self.fecg_index_list = list(range(1, num_mecg + 1)), list(range(1, num_fecg + 1))
        self.mecg_rand_index_list, self.fecg_rand_index_list = list(range(NUM_MECG_RANDS)), list(range(NUM_FECG_RANDS))

        self.load_into_memory = False
        if load_into_memory:
            self.loaded_dataset = []
            for data_i in range(len(self.dataset)):
                self.loaded_dataset.append(self[data_i])

            self.load_into_memory = True

        self.load_type = load_type

    def __len__(self):
        if self.load_type == 'normal':
            return len(self.dataset)
        else:
            return len(self.dataset)//4

    def num_mecg_fecg(self, fnames : [str]):
        max_mecg, max_fecg = 0, 0
        for fname in fnames:
            fname_raw = fname[len(self.data_dir)+1:]
            max_mecg = max(max_mecg, int(fname_raw[4:7]))
            max_fecg = max(max_fecg, int(fname_raw[8:10]))

        return max_mecg - 1, max_fecg - 1

    def get_bad_keys(self, d):
        # returns the keys that should be removed from the loadmat
        keys_to_remove = []
        for k, v in d.items():
            if type(v) != np.ndarray or 'fname' in k:
                keys_to_remove.append(k)
            else:
                d[k] = v.astype('double')

        return keys_to_remove

    def load_into_dictionary(self, fname, extend=False):
        if extend:
            inp = scipy.io.loadmat(f'{self.data_dir}/{fname}')
        else:
            inp = scipy.io.loadmat(fname)

        for key in self.get_bad_keys(inp):
            inp.pop(key)

        inp['mecg_sig'], inp['fecg_sig'], inp['offset'] = scale_signals(inp['mecg_sig'], inp['fecg_sig'])
        inp['fetal_mask'] = calc_multi_channel_peak_mask(inp['fecg_sig'])
        inp['maternal_mask'] = calc_multi_channel_peak_mask(inp['mecg_sig'])

        assert inp['fetal_mask'].shape == inp['maternal_mask'].shape

        inp['mecg_stft'], inp['fecg_stft'] = stft_batch(inp['mecg_sig']), stft_batch(inp['fecg_sig'])

        # assert inp['mecg_stft'].shape[0] == 2

        if torch.isnan(inp['mecg_stft']).any() or torch.isnan(inp['mecg_stft']).any():
            print(inp)
            raise SystemError

        return inp

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.load_type == 'normal':
            return self.load_into_dictionary(self.dataset[idx])

        else:
            base_ecg_fname = self.dataset[idx][len(self.data_dir)+1:]
            nums = base_ecg_fname.split('_')
            mecg_num, fecg_num, mecg_rand_window, fecg_rand_window = int(nums[1]), int(nums[2]), int(nums[3]), int(nums[4])
            diff_mecg = np.random.choice(self.mecg_index_list[:mecg_num-1] + self.mecg_index_list[mecg_num:])
            diff_fecg = np.random.choice(self.fecg_index_list[:fecg_num - 1] + self.fecg_index_list[fecg_num:])
            diff_window_fecg = np.random.choice(self.fecg_rand_index_list[:fecg_rand_window] + self.fecg_rand_index_list[fecg_rand_window+1:])

            base_dict = self.load_into_dictionary(base_ecg_fname, extend=True)
            diff_window_dict = self.load_into_dictionary(f'ecg_{mecg_num:03}_{fecg_num:02}_{mecg_rand_window:02}_{diff_window_fecg:02}', extend=True)
            diff_mecg_dict = self.load_into_dictionary(f'ecg_{diff_mecg:03}_{fecg_num:02}_{mecg_rand_window:02}_{fecg_rand_window:02}', extend=True)
            diff_fecg_dict = self.load_into_dictionary(f'ecg_{mecg_num:03}_{diff_fecg:02}_{mecg_rand_window:02}_{fecg_rand_window:02}', extend=True)

            # base == diff_mecg ~= diff_fecg_window != diff_fecg
            return {'base' : base_dict, 'diff_fecg_window' : diff_window_dict, 'diff_mecg' : diff_mecg_dict, 'diff_fecg' : diff_fecg_dict}

class ECGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, window_size, dataset_type: str = None, batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_DATA_WORKERS, pin_memory: bool = False):
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
            persistent_workers=True,
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
            drop_last=DROP_LAST,
            persistent_workers=True
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


if __name__ == '__main__':
    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, num_workers=0)
    dataloader = dm.test_dataloader()

    for i, d in enumerate(dataloader):
        SAMPLE_ECG = d
        break

    import pickle as pkl

    with open('sample_ecg.pkl', 'wb') as f:
        pkl.dump(SAMPLE_ECG, f)

