from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import numpy as np
from numpy.random import uniform
import torch
from pytorch_lightning import LightningDataModule
from hyperparams import *
from transforms import *
from wfdb.io import rdrecord

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

        elif split == 'competition':
            print('creating comp set')
            self.create_competition_dataset()
            self.load_type = 'competition'

        elif self.load_type == 'whole':
            self.create_whole_dataset()

        self.transforms = self.create_transforms()


    def __len__(self):
        if self.load_type == 'ss':
            return len(self.dataset) // 4
        else:
            return len(self.dataset)

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

    def create_competition_dataset(self):
        # TODO: remove hardcoding of data path
        self.dataset = [i for i in range(1, 76)]
        self.dataset.remove(10)

    def num_mecg_fecg(self, fnames : [str]):
        max_mecg, max_fecg = 0, 0
        for fname in fnames:
            fname_raw = fname[len(self.data_dir)+1:]
            max_mecg = max(max_mecg, int(fname_raw[4:7]))
            max_fecg = max(max_fecg, int(fname_raw[8:10]))

        return max_mecg - 1, max_fecg - 1

    def create_transforms(self):
        transforms = Transforms()

        transforms.add_transform('remove_bad_keys', None)

        # TODO: select random window from signal

        if self.load_type == 'competition':
            transforms.add_transform('filter', ('mecg_sig', 125, 1, 55, 3))
            transforms.add_transform('filter', ('fecg_sig', 125, 1, 55, 3))
            desired_length = WINDOW_LENGTH * NUM_WINDOWS
            transforms.add_transform('perform_trim', (desired_length, 0, 'mecg_sig', 'fecg_sig'))
            transforms.add_transform('trim_peaks', (desired_length, 0))
            transforms.add_transform('duplicate_keys', ('mecg_sig', 'binary_maternal_mask', 'binary_fetal_mask', 'noise'))
            transforms.add_transform('check_nans', ('mecg_sig', 'fecg_sig'))
            transforms.add_transform('reshape_keys', ('mecg_sig', 'fecg_sig', 'binary_fetal_mask',
                                                      'binary_maternal_mask', 'noise'))
            transforms.add_transform('reshape_peaks', ('fecg_peaks',))
            # transforms.add_transform('print_keys', ('fecg_sig', 'mecg_sig', 'binary_fetal_mask', 'noise'))
            transforms.add_transform('scale_multiple_segments', None)
            return transforms

        if self.load_type == 'whole':
            transforms.add_transform('downsample', ('fecg_sig', 2))
            transforms.add_transform('add_noise_signal', ('mecg_sig', 'mecg_sig'))
            transforms.add_transform('filter', ('mecg_sig', 125, 1, 50, 3))
            desired_length = WINDOW_LENGTH * NUM_WINDOWS
            desired_length_trim = int(WINDOW_LENGTH * NUM_WINDOWS * 1.5)
            transforms.add_transform('perform_trim', (desired_length_trim, 50, 'mecg_sig', 'fecg_sig', 'noise'))
            transforms.add_transform('trim_peaks', (desired_length_trim, 50))
            transforms.add_transform('resample', ('mecg_sig', None, None, desired_length, COMPRESS_RATIO))
            transforms.add_transform('resample', ('fecg_sig', 'noise', 'fecg_peaks', desired_length, COMPRESS_RATIO))
            transforms.add_transform('correct_peaks', (10, 'fecg_peaks', 'fecg_sig'))
            transforms.add_transform('get_signal_masks', ('fetal_mask', 'binary_fetal_mask', 'fecg_sig', 'fecg_peaks'))
            transforms.add_transform('get_signal_masks', ('maternal_mask', 'binary_maternal_mask', 'mecg_sig', None))
            transforms.add_transform('pop_keys', ('maternal_mask', 'fetal_mask'))
            transforms.add_transform('check_signal_shape', ('fecg_sig', 'mecg_sig'))
            transforms.add_transform('check_nans', ('fecg_sig', 'mecg_sig', 'fecg_peaks', 'binary_maternal_mask',
                                                     'binary_fetal_mask'))
            transforms.add_transform('reshape_keys', ('mecg_sig', 'fecg_sig', 'binary_fetal_mask', 'binary_maternal_mask', 'noise'))
            transforms.add_transform('reshape_peaks', ('fecg_peaks',))
            transforms.add_transform('scale_multiple_segments', None)
            return transforms

        if self.load_type == 'new':
            transforms.add_transform('transform_keys', None)

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

        self.transforms.perform_transforms(inp)

        inp['fname'] = fname

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

        self.transforms.perform_transforms(signal)

        signal['fname'] = (fecg_fname, mecg_fname)

        return signal

    def load_competition_item(self, idx):
        fname = self.dataset[idx]
        channel = COMPETITION_CHANNELS[idx]

        sig = rdrecord(f'Data/competition/set-a/a{fname:02}').p_signal[:, channel][np.newaxis, ...]
        with open(f'Data/competition/set-a-text/a{fname:02}.fqrs.txt', 'r') as f:
            peaks = np.array(f.read().split('\n')[:-1], dtype=np.int32)

        signal = {}
        signal['mecg_sig'] = ss.decimate(sig, 8, axis=-1) / 2
        signal['fecg_peaks'] = peaks / 8
        signal['fecg_sig'] = signal['mecg_sig'].copy()

        self.transforms.perform_transforms(signal)

        arr_copies = {k : v.copy() for k, v in signal.items() if 'array' in str(type(v))}

        signal.update(arr_copies)

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

        elif self.load_type == 'competition':
            return self.load_competition_item(idx)


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
            drop_last=DROP_LAST
        )

    def competition_dataloader(self):
        data = ECGDataset(
            data_dir = self.data_dir,
            window_size=self.window_size,
            load_type=self.load_type,
            numtaps=self.num_taps,
            split='competition'
        )

        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=DROP_LAST
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

