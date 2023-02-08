import os
import configparser
import random

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
from scipy.signal import decimate, resample

config = configparser.ConfigParser()
config.read('config.ini')

PROJECT_DIR = config.get('PATHS', 'PROJECT_PATH')
PAIRED_DATA_DIR = os.path.join(PROJECT_DIR, 'data/paired')
PAIRED_TRAIN_DIR = os.path.join(PROJECT_DIR, 'data/paired/train')
PAIRED_VALID_DIR = os.path.join(PROJECT_DIR, 'data/paired/validation')
PAIRED_DATA_DIR2 = os.path.join(PROJECT_DIR, 'data/paired2')
PAIRED_TRAIN_DIR2 = os.path.join(PROJECT_DIR, 'data/paired2/train')
PAIRED_VALID_DIR2 = os.path.join(PROJECT_DIR, 'data/paired2/validation')
INTERIM_MECG_DATA_DIR = os.path.join(PROJECT_DIR, 'data/interim/mecg')
INTERIM_FECG_DATA_DIR = os.path.join(PROJECT_DIR, 'data/interim/fecg')

SYNDB_FECG_DIR = os.path.join(INTERIM_FECG_DATA_DIR, 'fecgsyndb-1.0.0')
SYNDB_MECG_DIR = os.path.join(INTERIM_MECG_DATA_DIR, 'fecgsyndb-1.0.0')
BIDMC_MECG_DIR = os.path.join(INTERIM_MECG_DATA_DIR, 'bidmc-1.0.0')
FANTASIA_MECG_DIR = os.path.join(INTERIM_MECG_DATA_DIR, 'fantasia-1.0.0')
NSRDB_MECG_DIR = os.path.join(INTERIM_MECG_DATA_DIR, 'nsrdb-1.0.0')
STDB_MECG_DIR = os.path.join(INTERIM_MECG_DATA_DIR, 'stdb-1.0.0')

os.makedirs(PAIRED_DATA_DIR, exist_ok=True)
os.makedirs(PAIRED_TRAIN_DIR, exist_ok=True)
os.makedirs(PAIRED_VALID_DIR, exist_ok=True)

TOTAL_FILES = 40000
TARGET_FS = 125

SIGNAL_KEYS = ('filtered', 'light', 'raw')

def downsample(curr_freq : int, signal : np.ndarray, target_freq = 125, axis : int = 0) -> np.ndarray:
    '''Downsamples signal at curr_freq to target_freq '''
    if curr_freq % target_freq: # if need .resample
        target_samples = int(signal.shape[axis] / curr_freq * target_freq)
        return resample(signal, target_samples, axis=axis)

    return decimate(signal, int(curr_freq / target_freq), axis=axis)

def load_file(filename):
    d = loadmat(filename)
    return d

def amend_train():
    for f in os.listdir(PAIRED_TRAIN_DIR):
        prefix = f.partition('_')[0]
        c = f.split('_')[-1]
        d = loadmat(os.path.join(PAIRED_TRAIN_DIR, f))
        interval_start = d['fecg_start'].flatten()[0]
        fecg = load_file(d['fecg_file'][0])
        fecg_clean = downsample(250, fecg['filtered'][int(c)])[interval_start:interval_start+500]
        d['fecg_file'] = '/'.join(d['fecg_file'][0].split('/')[4:])
        d['mecg_file'] = '/'.join(d['mecg_file'][0].split('/')[4:])
        d['fecg_clean'] = fecg_clean.flatten()
        savemat(os.path.join(PAIRED_TRAIN_DIR2, f), d)
    return True

amend_train()

