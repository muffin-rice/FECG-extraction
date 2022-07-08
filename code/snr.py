import os

import numpy as np
import wfdb

import matplotlib.pyplot as plt

from load_data import ECGDataModule
from vae_train import DATA_DIR, NUM_DATA_WORKERS, BATCH_SIZE, NUM_TRAINER_WORKERS


def load_ecg_from_bidmc(fname : str) -> np.ndarray:
    samp = wfdb.rdsamp(fname)
    return samp[0][:,samp[1]['sig_name'].index('II,')]


def load_data_syndb(data_dir: str = 'Data/fecgsyndb-1.0.0') -> np.ndarray:
    '''returns a 10 (patients) x 5 (snr levels) x y (physiological levels, status) x 5 (repetition) x 75000 (250 Hz x 5 min) x 34 (channels)'''
    x1 = []
    for patient in range(1, 11):  # patient number
        x2 = []
        for snr in range(5):  # noise level

            x3 = []
            patient_noise_dir = f'{data_dir}/sub{patient:02}/snr{snr * 3:02}dB'
            fnames = [f for f in os.listdir(patient_noise_dir)]  # fnames is unsorted, should sort x3 after
            fnames_trunc = set([fname[:-4] for fname in fnames])
            seen_fnames = set()

            for fname in fnames_trunc:  # for all fnames, may be of any trial
                if f'{fname[:13]}{fname[17:]}' in seen_fnames:  # if same combination of diff trial was seen
                    continue

                seen_fnames.add(f'{fname[:13]}{fname[17:]}')  # add current combination
                x4 = []  # list of all the trials
                for i in range(1, 6):  # for each trial
                    x4.append(f'{patient_noise_dir}/{fname[:13]}_l{i}_{fname[17:]}')

                assert len(x4) == 5

                x3.append(x4)  # x4 is sorted by trial

            x2.append(sorted(x3, key=lambda x: x[0]))  # sort x3

        x1.append(x2)

    return np.array(x1, dtype=object)



if __name__ == '__main__':
    dm = ECGDataModule(data_dir=DATA_DIR, window_size=500, dataset_type='', num_workers=NUM_DATA_WORKERS,
                         batch_size=max(2, int(BATCH_SIZE / NUM_TRAINER_WORKERS)))

    snrs = []
    dl = dm.train_dataloader()
    for d in dl:
        snrs.extend(d['snr'])
    print(len(snrs))
    print("Train calculated")
    plt.hist(snrs)
    plt.show()

    dl = dm.val_dataloader()
    for d in dl:
        snrs.extend(d['snr'])
    print("Val calculated")
    plt.hist(snrs)
    plt.show()

    dl = dm.test_dataloader()
    for d in dl:
        snrs.extend(d['snr'])
    print("Test calculated")
    plt.hist(snrs)
    plt.show()

    np.save("/Users/Richard/git/FECG-extraction/run/snrs.npy", snrs)
