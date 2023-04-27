import numpy as np
from utils2 import scale_signals, generate_gaussian_noise_by_shape, get_random_mfratio, resample_signal_noise_peak, \
    return_peaks, gauss_kernel, correct_peaks
from hyperparams import MF_RATIO, MF_RATIO_STD, WINDOW_LENGTH, PEAK_SCALE, PEAK_SIGMA, BINARY_PEAK_WINDOW, \
    NOISE, PAD_LENGTH
from scipy import signal as ss
from scipy.signal import filtfilt, butter
from numpy import random
from torch import from_numpy
import torch

def calc_peak_mask(sig : np.array, num_windows : int, peak_window = PEAK_SCALE, peak_sigma = PEAK_SIGMA,
                   binary_peak_window = BINARY_PEAK_WINDOW, actual_peaks = None,):
    # signal should be single channel; 1 x sig_length

    if actual_peaks is None:
        peaks = correct_peaks(return_peaks(sig), sig, window_radius=5)
    else:
        peaks = correct_peaks(actual_peaks, sig, window_radius=5)

    peak_mask = np.zeros(sig.shape)

    binary_peak_mask = np.zeros(sig.shape)

    for peak in peaks:
        peak = int(peak)
        if peak >= len(sig):
            break
        left, right = max(peak-int(peak_window/2), 0), min(peak+int(peak_window/2)+1, len(sig))
        peak_mask[left : right] = gauss_kernel(peak_window, peak_sigma)[:right-left] # kernel slightly bugged on edges

        binary_peak_mask[peak] = 1

    return peak_mask, binary_peak_mask

def calc_multi_channel_peak_mask(sig : np.ndarray, num_windows : int, peak_window = PEAK_SCALE, peak_sigma = PEAK_SIGMA,
                                 actual_peaks = None):
    mask = np.zeros(sig.shape)
    binary_peak_mask = np.zeros(sig.shape)

    for channel in range(sig.shape[0]):
        mask[channel, :], binary_peak_mask[channel, :] = calc_peak_mask(sig[channel, :], peak_window, peak_sigma,
                                                                        actual_peaks=actual_peaks)

    return mask, binary_peak_mask

def get_bad_keys(d):
    # returns the keys that should be removed from the loadmat
    keys_to_remove = []
    for k, v in d.items():
        if type(v) != np.ndarray or 'file' in k or 'fname' in k:
            keys_to_remove.append(k)
        else:
            d[k] = v.astype(np.float32)

    return keys_to_remove

class Transforms:
    def __init__(self):
        self.transforms = []
        self.transform_dictionary = {
            'add_brownian_noise': self.add_brownian_noise,
            'add_noise_signal': self.add_noise_signal,
            'add_to_dict': self.add_to_dict,
            'assert_nonzero': self.assert_nonzero,
            'assert_nonzero2': self.assert_nonzero2,
            'change_dtype': self.change_dtype,
            'check_nans': self.check_nans,
            'check_signal_shape': self.check_signal_shape,
            'correct_peaks': self.correct_peaks,
            'downsample': self.downsample,
            'duplicate_keys': self.duplicate_keys,
            'filter': self.filter,
            'get_signal_masks': self.get_signal_masks,
            'perform_trim': self.perform_trim,
            'pop_keys': self.pop_keys,
            'print_keys': self.print_keys,
            'remove_bad_keys': self.remove_bad_keys,
            'reshape_keys': self.reshape_keys,
            'reshape_peaks': self.reshape_peaks,
            'resample': self.resample_signal,
            'scale_segment' : self.scale_segment,
            'scale_multiple_segments' : self.scale_multiple_segments,
            'suppress_peaks' : self.mute_fecg_peak,
            'transform_keys' : self.perform_transforms,
            'vary_signal_strength' : self.vary_signal_strength,
        }

    def add_transform(self, transform_name : str, transform_params : (any,)):
        if transform_name in self.transform_dictionary:
            self.transforms.append((self.transform_dictionary[transform_name], transform_params))
        else:
            raise NotImplementedError(f'Transform {transform_name} is not registered inside transform dict')

    def perform_transforms(self, signal_dict):
        for transform, transform_params in self.transforms:
            if transform_params:
                transform(signal_dict, *transform_params)
            else:
                transform(signal_dict)

    def transform_keys(self, signal_dict):
        '''transforms all of the keys'''
        signal_dict['fecg_sig'] = signal_dict['fecg_clean']
        signal_dict['mecg_sig'] = signal_dict['mecg_clean']
        signal_dict['fecg_peaks'] = signal_dict['fecg_peaks'][0].astype(int)
        signal_dict['noise'] = signal_dict['mecg_signal'] - signal_dict['mecg_clean']

    def downsample(self, signal_dict, key, rate):
        signal_dict[key] = ss.resample(signal_dict[key][0, :], signal_dict[key].shape[1] // rate)[np.newaxis, :]

    def filter(self, signal_dict, key, fs, lowcut, highcut, order):
        nyquist = fs/2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='bandpass', analog=False, output='ba')
        signal_dict[key] = filtfilt(b, a, signal_dict[key])

    def remove_bad_keys(self, signal_dict):
        '''remove the irrelevant keys generated by scipy's loadmat'''
        for key in get_bad_keys(signal_dict):
            signal_dict.pop(key)

    def duplicate_keys(self, signal_dict, key_to_dupe, *keys):
        for key in keys:
            signal_dict[key] = np.zeros_like(signal_dict[key_to_dupe])

    def perform_trim(self, signal_dict, desired_length, *key_groups):
        '''trim if necessary to reduce computational load on resampling
        also takes a random window for the trim'''
        # key group where the trim is the same
        for key_group in key_groups:
            signal_length = signal_dict[key_group[0]].shape[1]
            # TODO: do not assert
            assert signal_length > desired_length, f'signal length is not long enough {signal_length}'

            random_start = random.randint(signal_length - desired_length)
            for key in key_group:
                if 'peak' in key:
                    signal_dict[key] = signal_dict[key][(signal_dict[key] < desired_length+random_start) &
                                                              (random_start < signal_dict[key])].astype(int) - random_start

                else:
                    signal_dict[key] = signal_dict[key][:, random_start:random_start + desired_length]

    def resample_signal(self, signal_dict, resample_key, noise_key, peak_key, desired_length, ratio):
        if noise_key is None and peak_key is None:
            signal_dict[resample_key], _, _ = resample_signal_noise_peak(
                signal_dict[resample_key], ratio=ratio, desired_length=desired_length)
        else:
            signal_dict[resample_key], signal_dict[noise_key], signal_dict[peak_key] = resample_signal_noise_peak(
                signal_dict[resample_key], ratio=ratio, desired_length=desired_length, noise=signal_dict[noise_key],
                peak = signal_dict[peak_key])

    def get_signal_masks(self, signal_dict, mask_key, binary_mask_key, sig_key, peak_key):
        '''puts signal mask keys into dict based on the given signal key'''
        if peak_key is None:
            signal_dict[mask_key], signal_dict[binary_mask_key] = calc_multi_channel_peak_mask(signal_dict[sig_key],
                                                                                               signal_dict['num_windows'])
        else:
            signal_dict[mask_key], signal_dict[binary_mask_key] = calc_multi_channel_peak_mask(signal_dict[sig_key],
                                                                                               signal_dict['num_windows'],
                                                                                               actual_peaks=
                                                                                               signal_dict[peak_key],)

    def reshape_keys(self, signal_dict, *keys):
        '''reshape signal into n_segments x segment for individual scaling'''
        for key in keys:
            signal_dict[key] = signal_dict[key].reshape(signal_dict['num_windows'], WINDOW_LENGTH)

    def reshape_peaks(self, signal_dict, peak_key, pad_length=PAD_LENGTH):
        peak_locs = np.zeros((signal_dict['num_windows'], pad_length))
        for window in range(signal_dict['num_windows']):
            borders = (window*WINDOW_LENGTH, (window+1)*WINDOW_LENGTH)
            between = np.logical_and(signal_dict[peak_key] > borders[0], signal_dict[peak_key] < borders[1])
            bordered_peaks = (signal_dict[peak_key][between] - borders[0]) / WINDOW_LENGTH
            pad_amount = pad_length-bordered_peaks.shape[0]
            assert pad_amount >= 0
            # assert pad_amount < pad_length, f'peaks are zero: {signal_dict[peak_key]} on window {window} and len {WINDOW_LENGTH}'
            padded_peaks = np.pad(bordered_peaks, pad_width=(0, pad_amount), mode='constant')
            peak_locs[window,:] = padded_peaks

        signal_dict[peak_key] = peak_locs

    def check_signal_shape(self, signal_dict, key1, key2):
        assert signal_dict[key1].shape == signal_dict[key2].shape, f'{signal_dict[key1].shape} {signal_dict[key2].shape}'

    def check_nans(self, signal_dict, *keys_to_check):
        for key in keys_to_check:
            assert not np.isnan(signal_dict[key]).any(), f'Signals are nan {signal_dict[key]}'

    def add_noise_signal(self, signal_dict, noise, noise_key, shape_key):
        '''adds a gaussian noise (random) '''
        signal_dict[noise_key] += generate_gaussian_noise_by_shape(shape=signal_dict[shape_key].shape, stdev=noise)

    def scale_segment(self, signal_dict):
        mf_ratio = get_random_mfratio(MF_RATIO, MF_RATIO_STD)

        signal_dict['mf_ratio'] = mf_ratio

        signal_dict['mecg_sig'], signal_dict['fecg_sig'], signal_dict['offset'] = scale_signals(signal_dict['mecg_sig'],
                                                                                                signal_dict['fecg_sig'],
                                                                                                mf_ratio,
                                                                                                signal_dict['noise'])

    def correct_peaks(self, signal_dict, window_radius, peak_key, sig_key):
        signal_dict[peak_key] = correct_peaks(peaks = signal_dict[peak_key], sig = signal_dict[sig_key][0],
                                              window_radius=window_radius)

    def scale_multiple_segments(self, signal_dict, mf_ratio_mean, mf_ratio_std):
        '''scales multiple segments'''
        mf_ratio = get_random_mfratio(mf_ratio_mean, mf_ratio_std)
        signal_dict['offset'] = np.zeros_like(signal_dict['mecg_sig'])
        signal_dict['mf_ratio'] = mf_ratio
        for i in range(signal_dict['num_windows']):
            signal_dict['mecg_sig'][i, :], signal_dict['fecg_sig'][i, :], signal_dict['offset'][i, :] = scale_signals(
                signal_dict['mecg_sig'][[i], :],
                signal_dict['fecg_sig'][[i], :], mf_ratio,
                signal_dict['noise'][[i], :])

    def pop_keys(self, signal_dict, *keys_to_pop):
        for key in keys_to_pop:
            signal_dict.pop(key)

    def change_dtype(self, signal_dict, dtype, *keys_to_change):
        if keys_to_change[0] is None:
            keys_to_change = signal_dict.keys()

        for key in keys_to_change:
            if 'arr' in str(type(signal_dict[key])):
                signal_dict[key] = from_numpy(signal_dict[key].copy()).to(dtype)
            else:
                signal_dict[key] = torch.tensor(signal_dict[key], dtype=dtype)

    def add_to_dict(self, signal_dict, key, val):
        signal_dict[key] = val

    def print_keys(self, signal_dict, *keys_to_print):
        for key in keys_to_print:
            print(f'{key} : {signal_dict[key]}')

    def assert_nonzero(self, signal_dict, *keys):
        for key in keys:
            assert signal_dict[key].any()

    def assert_nonzero2(self, signal_dict, *keys):
        for key in keys:
            assert signal_dict[key].any(axis=1).all()

    def add_brownian_noise(self, signal_dict, stdev, noise_key):
        random_noise = generate_gaussian_noise_by_shape(signal_dict[noise_key].shape, stdev)
        signal_dict[noise_key] += np.cumsum(random_noise)

    def mute_fecg_peak(self, signal_dict, chance, max_peaks, window_rad):
        if chance == 0 or max_peaks == 0:
            return
        peak_key = 'fecg_peaks'
        noise_key = 'noise'
        sig_key = 'fecg_sig'
        peaks_to_cancel = []
        for peak in signal_dict[peak_key]:
            if random.random() < chance:
                peaks_to_cancel.append(peak)

                if len(peaks_to_cancel) > max_peaks:
                    break

        for cancel_peak in peaks_to_cancel:
            signal_dict[noise_key][0, cancel_peak - window_rad : cancel_peak + window_rad] -= \
                signal_dict[sig_key][0, cancel_peak - window_rad : cancel_peak + window_rad]

    def vary_signal_strength(self, signal_dict, signal_to_vary, strength_std):
        vary_length = signal_dict[signal_to_vary].shape[-1]

        vary = 1 + generate_gaussian_noise_by_shape(vary_length, strength_std)

        signal_dict[signal_to_vary] *= vary