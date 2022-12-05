from torch.utils.data import DataLoader, Dataset
import scipy.io
import scipy
import numpy as np
from numpy.random import uniform
import torch
import os
from os.path import isfile
from pytorch_lightning import LightningDataModule
from hyperparams import *
from utils2 import scale_signals, stft_batch, invert_stft_batch, filt, return_peaks, gauss_kernel, \
    generate_gaussian_noise_by_shape, get_random_mfratio, resample_arr, correct_peaks


def load_data(data_dir: str) -> np.array:
    dataset = []
    fnames = os.listdir(data_dir)
    for fname in fnames:
        true_fname = f'{data_dir}/{fname}'
        if fname != '.DS_Store':
            dataset.append(true_fname)

    # train_data, test_val_data = train_test_split(dataset, test_size=0.5)
    # test_data, val_data = train_test_split(test_val_data, test_size=0.5)

    return np.array(dataset, dtype=object)

def calc_peak_mask(sig : np.array, peak_window = PEAK_SCALE, peak_sigma = PEAK_SIGMA,
                   binary_peak_window = BINARY_PEAK_WINDOW, actual_peaks = None):
    # signal should be single channel; 1 x sig_legnth

    if actual_peaks is None:
        peaks = correct_peaks(return_peaks(sig[:]), sig, window_len=3)
    else:
        peaks = correct_peaks(actual_peaks, sig, window_len=3)

    peak_mask = np.zeros(sig.shape)

    binary_peak_mask = np.zeros(sig.shape)

    for peak in peaks:
        peak = int(peak)
        if peak >= 500:
            break
        left, right = peak-int(peak_window/2), peak+int(peak_window/2)+1
        peak_mask[left : right] = gauss_kernel(peak_window, peak_sigma)[:len(peak_mask[left : right])]

        binary_peak_mask[peak] = 1

        # if peak < binary_peak_window:
        #     binary_peak_mask[0:peak+binary_peak_window]= 1
        # elif peak + binary_peak_window >= binary_peak_mask.shape[0]:
        #     binary_peak_mask[peak-binary_peak_window:] = 1
        # else:
        #     binary_peak_mask[peak-binary_peak_window:peak+binary_peak_window] = 1

    return peak_mask, binary_peak_mask

def calc_multi_channel_peak_mask(sig : np.ndarray, peak_window = PEAK_SCALE, peak_sigma = PEAK_SIGMA, actual_peaks = None):
    mask = np.zeros(sig.shape)
    binary_peak_mask = np.zeros(sig.shape)

    for channel in range(sig.shape[0]):
        mask[channel, :], binary_peak_mask[channel, :] = calc_peak_mask(sig[channel, :], peak_window, peak_sigma,
                                                                        actual_peaks=actual_peaks)

    return mask, binary_peak_mask


class ECGDataset(Dataset):
    def __init__(self, window_size, data_dir=DATA_DIR, dataset_type: str = None, split: str = 'train',
                 load_into_memory=LOAD_INTO_MEMORY, load_type : str = 'normal', noise = NOISE, ratio = COMPRESS_RATIO):
        super(ECGDataset, self).__init__()
        # self.train_data, self.val_data, self.test_data = load_data(data_dir)
        if TRAIN_PEAKHEAD:
            self.data_dir = f'Data/competition/{split}'
        else:
            self.data_dir = f'{data_dir}/{split}'
        self.dataset = load_data(f'{data_dir}/{split}')
        self.noise = noise

        self.window_size = window_size  # window size

        if load_type == 'ss':
            num_mecg, num_fecg = self.num_mecg_fecg(self.dataset)
            # index starts at 1
            self.mecg_index_list, self.fecg_index_list = list(range(1, num_mecg + 1)), list(range(1, num_fecg + 1))
            self.mecg_rand_index_list, self.fecg_rand_index_list = list(range(NUM_MECG_RANDS)), list(range(NUM_FECG_RANDS))

        self.load_type = load_type
        self.ratio = ratio

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
            if type(v) != np.ndarray or 'file' in k or 'fname' in k:
                keys_to_remove.append(k)
            else:
                d[k] = v.astype(np.float32)

        return keys_to_remove

    def transform_keys(self, d):
        d2 = {}
        d2['fecg_sig'] = d['fecg_signal']
        d2['mecg_sig'] = d['mecg_clean']
        d2['gt_fecg_sig'] = d['fecg_clean']
        d2['fecg_peaks'] = d['fecg_peaks'][0]

        return d2

    def load_into_dictionary(self, fname, extend=False):
        try:
            if extend:
                inp = scipy.io.loadmat(f'{self.data_dir}/{fname}')
            else:
                inp = scipy.io.loadmat(fname)
        except FileNotFoundError:
            print(f'{fname} not found')
            raise FileNotFoundError

        for key in self.get_bad_keys(inp):
            inp.pop(key)

        if self.load_type == 'new':
            inp = self.transform_keys(inp)

        # inp['noise'] = generate_noise_by_shape(shape=inp['fecg_sig'].shape, stdev=NOISE_STD)
        # noisy_mecg = inp['mecg_sig'] + inp['noise']

        # randomly up/downsample mecg
        resample_target = int(len(inp['fecg_sig'].T) * uniform(low = self.ratio[0], high = self.ratio[1])) # ratios are 500/600 ==> oppos
        inp['mecg_sig'] = resample_arr(inp['mecg_sig'].T, resample_target)[:500, 0:].T

        # randomly up/downsample fecg
        resample_ratio = uniform(low=self.ratio[0], high=self.ratio[1])
        resample_target = int(len(inp['fecg_sig'].T) * resample_ratio)  # ratios are 500/600 ==> oppos
        inp['fecg_sig'] = resample_arr(inp['fecg_sig'].T, resample_target)[:500, 0:].T
        inp['gt_fecg_sig'] = resample_arr(inp['gt_fecg_sig'].T, resample_target)[:500, 0:].T
        inp['fecg_peaks'] = (resample_ratio * inp['fecg_peaks']).astype(int)

        assert not (torch.isnan(torch.from_numpy(inp['fecg_sig'])).any() or
                    torch.isnan(torch.from_numpy(inp['mecg_sig'])).any()), f'Signals are nan {inp}'

        # calculate noise first so scaling will occur on mecg sig + noise
        inp['noise'] = generate_gaussian_noise_by_shape(shape=inp['fecg_sig'].shape, stdev=self.noise)

        # scale and normalize inp['fecg_sig'] and inp['mecg_sig']
        mf_ratio = get_random_mfratio(MF_RATIO, MF_RATIO_STD)
        inp['mecg_sig'] += inp['noise'].numpy()
        _, inp['gt_fecg_sig'], _ = scale_signals(inp['mecg_sig'], inp['gt_fecg_sig'], mf_ratio)
        inp['mecg_sig'], inp['fecg_sig'], inp['offset'] = scale_signals(inp['mecg_sig'], inp['fecg_sig'], mf_ratio)

        inp['fetal_mask'], inp['binary_fetal_mask'] = calc_multi_channel_peak_mask(inp['fecg_sig'], actual_peaks = inp['fecg_peaks'])
        inp['maternal_mask'], inp['binary_maternal_mask'] = calc_multi_channel_peak_mask(inp['mecg_sig'])

        assert inp['fetal_mask'].shape == inp['maternal_mask'].shape

        inp['fname'] = fname
        inp.pop('fecg_peaks') # binary_fetal_mask instead (stacks better)

        return inp

    def __getitem__(self, idx, noise=NOISE):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.load_type in ('normal', 'new'):
            return self.load_into_dictionary(self.dataset[idx])

        elif self.load_type == 'ss':
            # self supervised; load four different signals and return all of them
            base_ecg_fname = self.dataset[idx][len(self.data_dir)+1:]
            nums = base_ecg_fname.split('_')
            # indexing weird because format ecg_x_y – x and y are 1-indexed.
            mecg_num, fecg_num, mecg_rand_window, fecg_rand_window = int(nums[1]), int(nums[2]), int(nums[3]), int(nums[4])
            diff_mecg = np.random.choice(self.mecg_index_list[:mecg_num-1] + self.mecg_index_list[mecg_num:])
            diff_fecg = np.random.choice(self.fecg_index_list[:fecg_num-1] + self.fecg_index_list[fecg_num:])
            diff_window_fecg = np.random.choice(self.fecg_rand_index_list[:fecg_rand_window] + self.fecg_rand_index_list[fecg_rand_window+1:])

            # "correct" dictionary
            base_dict = self.load_into_dictionary(base_ecg_fname, extend=True)
            try:
                # change the window; latent space diff should be small (SWITCH_FECG_WINDOW_ALPHA)
                diff_window_dict = self.load_into_dictionary(f'ecg_{mecg_num:03}_{fecg_num:02}_{mecg_rand_window:02}_{diff_window_fecg:02}', extend=True)
                # change the mecg; latent space diff should be 0 (FIX_FECG_ALPHA)
                diff_mecg_dict = self.load_into_dictionary(f'ecg_{diff_mecg:03}_{fecg_num:02}_{mecg_rand_window:02}_{fecg_rand_window:02}', extend=True)
                # change the fecg; latent space diff should be big (SWITCH_FECG_ALPHA)
                diff_fecg_dict = self.load_into_dictionary(f'ecg_{mecg_num:03}_{diff_fecg:02}_{mecg_rand_window:02}_{fecg_rand_window:02}', extend=True)

            except FileNotFoundError:
                print(base_ecg_fname)
                raise FileNotFoundError


            # base == diff_mecg ~= diff_fecg_window != diff_fecg
            return {'base' : base_dict, 'diff_fecg_window' : diff_window_dict, 'diff_mecg' : diff_mecg_dict, 'diff_fecg' : diff_fecg_dict}

class ECGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, window_size, dataset_type: str = None, batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_DATA_WORKERS, pin_memory: bool = False, load_type : str = LOAD_TYPE):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.load_type = load_type

    def train_dataloader(self):
        data = ECGDataset(
            data_dir=self.data_dir,
            window_size=self.window_size,
            dataset_type=self.dataset_type,
            load_type=self.load_type,
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
            load_type=self.load_type,
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
            data_dir=self.data_dir,
            window_size=self.window_size,
            dataset_type=self.dataset_type,
            load_type=self.load_type,
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

