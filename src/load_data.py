from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import numpy as np
from numpy.random import uniform
import torch
from pytorch_lightning import LightningDataModule
from hyperparams import *
from wfdb.processing import correct_peaks
from transforms import *

def load_data(data_dir: str) -> np.array:
    dataset = []
    fnames = os.listdir(data_dir)
    for fname in fnames:
        true_fname = f'{data_dir}/{fname}'
        if fname != '.DS_Store':
            dataset.append(true_fname)

    return np.array(dataset, dtype=object)


class ECGDataset(Dataset):
    def __init__(self, window_size, split: str = 'train', numtaps = NUM_TAPS, data_dir=DATA_DIR, load_type=LOAD_TYPE):
        super(ECGDataset, self).__init__()
        # self.train_data, self.val_data, self.test_data = load_data(data_dir)
        self.noise = NOISE
        self.window_size = window_size  # window size
        self.load_type = load_type
        self.ratio = COMPRESS_RATIO
        self.num_taps = numtaps

        if TRAIN_PEAKHEAD:
            # TODO: remove hardcoded
            self.data_dir = f'Data/competition/{split}'
        else:
            self.data_dir = f'{data_dir}/{split}'

        # TODO: change self.dataset function
        self.dataset = load_data(f'{data_dir}/{split}')

        if self.load_type == 'ss':
            self.create_ss_dataset()

        elif self.load_type == 'whole':
            self.create_whole_dataset()

        self.transforms = self.create_transforms()


    def __len__(self):
        if self.load_type in ('whole', 'normal'):
            return len(self.dataset)
        else:
            return len(self.dataset)//4

    def create_whole_dataset(self):
        '''dataset built on indices; load separately and then map later'''
        # TODO: remove hardcode
        self.fecg_dataset = load_data(f'Data/preprocessed_data/fecg_signal')
        self.mecg_dataset = load_data(f'Data/preprocessed_data/mecg_signal')

        import pickle as pkl
        with open(f'{self.data_dir}/index_mapping.pkl', 'rb') as f:
            self.dataset = pkl.load(f)

    def create_ss_dataset(self):
        num_mecg, num_fecg = self.num_mecg_fecg(self.dataset)
        # index starts at 1
        self.mecg_index_list, self.fecg_index_list = list(range(1, num_mecg + 1)), list(range(1, num_fecg + 1))
        self.mecg_rand_index_list, self.fecg_rand_index_list = list(range(NUM_MECG_RANDS)), list(range(NUM_FECG_RANDS))

    def num_mecg_fecg(self, fnames : [str]):
        max_mecg, max_fecg = 0, 0
        for fname in fnames:
            fname_raw = fname[len(self.data_dir)+1:]
            max_mecg = max(max_mecg, int(fname_raw[4:7]))
            max_fecg = max(max_fecg, int(fname_raw[8:10]))

        return max_mecg - 1, max_fecg - 1

    def create_transforms(self):
        transforms = []

        transforms.append(remove_bad_keys)

        if self.load_type == 'new':
            transforms.append(transform_keys)

        if self.load_type == 'whole': # resmaple with different parameters
            transforms.append(downsample_fecg)
            filterer = Filterer(numtaps=self.num_taps)
            transforms.append(filterer.perform_filter)
            resampler = Resampler(desired_length=WINDOW_LENGTH * NUM_WINDOWS, shape_index=1, ratio=self.ratio)
            transforms.append(resampler.perform_initial_trim)
            transforms.append(resampler.resample_signal)

        else:
            resampler = Resampler()
            transforms.append(resampler.resample_signal)

        transforms.append(get_signal_masks)
        transforms.append(check_signal_shapes)
        transforms.append(check_nans)
        transforms.append(add_noise_signal)

        if self.load_type == 'whole':
            transforms.append(split_signal_into_segments)
            transforms.append(scale_multiple_segments)
        else:
            transforms.append(scale_segment)

        return transforms

    def load_paired_item(self, fname, extend=False):
        '''loads a single signal from some fname that corresponds to a .mat file with
        the correct signals and peaks'''
        try:
            if extend:
                inp = loadmat(f'{self.data_dir}/{fname}')
            else:
                inp = loadmat(fname)
        except FileNotFoundError:
            print(f'{fname} not found')
            raise FileNotFoundError

        for transform in self.transforms:
            transform(inp)

        inp['fname'] = fname
        inp.pop('fecg_peaks') # peaks are in binary_fetal_mask instead (stackable)

        return inp

    def load_ss_item(self, idx):
        # self supervised; load four different signals and return all of them
        base_ecg_fname = self.dataset[idx][len(self.data_dir) + 1:]
        nums = base_ecg_fname.split('_')
        # indexing weird because format ecg_x_y – x and y are 1-indexed.
        mecg_num, fecg_num, mecg_rand_window, fecg_rand_window = int(nums[1]), int(nums[2]), int(nums[3]), int(nums[4])
        diff_mecg = np.random.choice(self.mecg_index_list[:mecg_num - 1] + self.mecg_index_list[mecg_num:])
        diff_fecg = np.random.choice(self.fecg_index_list[:fecg_num - 1] + self.fecg_index_list[fecg_num:])
        diff_window_fecg = np.random.choice(
            self.fecg_rand_index_list[:fecg_rand_window] + self.fecg_rand_index_list[fecg_rand_window + 1:])

        # "correct" dictionary
        base_dict = self.load_paired_item(base_ecg_fname, extend=True)
        try:
            # change the window; latent space diff should be small (SWITCH_FECG_WINDOW_ALPHA)
            diff_window_dict = self.load_paired_item(
                f'ecg_{mecg_num:03}_{fecg_num:02}_{mecg_rand_window:02}_{diff_window_fecg:02}', extend=True)
            # change the mecg; latent space diff should be 0 (FIX_FECG_ALPHA)
            diff_mecg_dict = self.load_paired_item(
                f'ecg_{diff_mecg:03}_{fecg_num:02}_{mecg_rand_window:02}_{fecg_rand_window:02}', extend=True)
            # change the fecg; latent space diff should be big (SWITCH_FECG_ALPHA)
            diff_fecg_dict = self.load_paired_item(
                f'ecg_{mecg_num:03}_{diff_fecg:02}_{mecg_rand_window:02}_{fecg_rand_window:02}', extend=True)

        except FileNotFoundError:
            print(base_ecg_fname)
            raise FileNotFoundError

        # base == diff_mecg ~= diff_fecg_window != diff_fecg
        return {'base': base_dict, 'diff_fecg_window': diff_window_dict, 'diff_mecg': diff_mecg_dict,
                'diff_fecg': diff_fecg_dict}

    def load_indexpair_item(self, index_pair):
        fecg_fname = self.fecg_dataset[index_pair[0]]
        mecg_fname = self.mecg_dataset[index_pair[1]]

        try:
            fecg = loadmat(fecg_fname)
            mecg = loadmat(mecg_fname)
        except FileNotFoundError:
            print(f'{fecg_fname} or {mecg_fname} not found')
            raise FileNotFoundError

        signal = {}
        signal['mecg_sig'] = mecg['mecg_signal']
        signal['fecg_sig'] = fecg['gt_fecg_sig']
        signal['fecg_peaks'] = fecg['fecg_peaks'].astype(int)
        signal['noise'] = fecg['fecg_sig'] - fecg['gt_fecg_sig']

        for transform in self.transforms:
            transform(signal)

        signal['fname'] = (fecg_fname, mecg_fname)
        signal.pop('fecg_peaks')  # peaks are in binary_fetal_mask instead (stackable)

        return signal

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.load_type in ('normal', 'new'):
            return self.load_paired_item(self.dataset[idx])

        elif self.load_type == 'ss':
            return self.load_ss_item(idx)

        elif self.load_type == 'whole':
            return self.load_indexpair_item(self.dataset[idx])

class ECGDataModule(LightningDataModule):
    def __init__(self, data_dir: str, window_size, dataset_type: str = None, batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_DATA_WORKERS, pin_memory: bool = False, load_type : str = LOAD_TYPE,
                 num_taps : int = NUM_TAPS):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.load_type = load_type
        self.num_taps = num_taps

    def train_dataloader(self):
        data = ECGDataset(
            data_dir=self.data_dir,
            window_size=self.window_size,
            load_type=self.load_type,
            numtaps=self.num_taps,
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
            load_type=self.load_type,
            numtaps=self.num_taps,
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
            load_type=self.load_type,
            numtaps=self.num_taps,
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
    '''main function creates the sample_ecg file'''
    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, num_workers=1)
    dataloader = dm.val_dataloader()

    for i, d in enumerate(dataloader):
        SAMPLE_ECG = d
        break

    import pickle as pkl

    with open('sample_ecg.pkl', 'wb') as f:
        pkl.dump(SAMPLE_ECG, f)

