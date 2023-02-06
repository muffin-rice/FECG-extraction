import numpy as np
from utils2 import scale_signals, generate_gaussian_noise_by_shape, get_random_mfratio, resample_signal_noise_peak, \
    return_peaks, gauss_kernel, correct_peaks
from hyperparams import MF_RATIO, NUM_WINDOWS, MF_RATIO_STD, WINDOW_LENGTH, PEAK_SCALE, PEAK_SIGMA, BINARY_PEAK_WINDOW, \
    NOISE, PAD_LENGTH
from math import cos, sqrt, pi, sin
from scipy import signal as ss
from scipy.signal import filtfilt, butter
from torchaudio.functional import bandpass_biquad
from torch import from_numpy

def calc_peak_mask(sig : np.array, peak_window = PEAK_SCALE, peak_sigma = PEAK_SIGMA,
                   binary_peak_window = BINARY_PEAK_WINDOW, actual_peaks = None):
    # signal should be single channel; 1 x sig_length

    if actual_peaks is None:
        peaks = correct_peaks(return_peaks(sig), sig, window_radius=5)
    else:
        peaks = correct_peaks(actual_peaks, sig, window_radius=5)

    peak_mask = np.zeros(sig.shape)

    binary_peak_mask = np.zeros(sig.shape)

    for peak in peaks:
        peak = int(peak)
        if peak >= WINDOW_LENGTH*NUM_WINDOWS:
            break
        left, right = peak-int(peak_window/2), peak+int(peak_window/2)+1
        peak_mask[left : right] = gauss_kernel(peak_window, peak_sigma)[:len(peak_mask[left : right])]

        binary_peak_mask[peak] = 1

    return peak_mask, binary_peak_mask

def calc_multi_channel_peak_mask(sig : np.ndarray, peak_window = PEAK_SCALE, peak_sigma = PEAK_SIGMA, actual_peaks = None):
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
        # filtering
        self.transforms = []

    def add_transform(self, transform_name : str, transform_params : (any,)):
        # TODO: use dict mapping instead of ifs
        if transform_name == 'transform_keys':
            self.transforms.append((self.perform_transforms, None))
        elif transform_name == 'downsample':
            self.transforms.append((self.downsample, transform_params))
        elif transform_name == 'filter':
            self.transforms.append((self.filter, transform_params))
        elif transform_name == 'remove_bad_keys':
            self.transforms.append((self.remove_bad_keys, None))
        elif transform_name == 'duplicate_fecg':
            self.transforms.append((self.duplicate_fecg, transform_params))
        elif transform_name == 'perform_trim':
            self.transforms.append((self.perform_trim, transform_params))
        elif transform_name == 'trim_peaks':
            self.transforms.append((self.trim_peaks, transform_params))
        elif transform_name == 'resample':
            self.transforms.append((self.resample_signal, transform_params))
        elif transform_name == 'get_signal_masks':
            self.transforms.append((self.get_signal_masks, transform_params))
        elif transform_name == 'reshape_keys':
            self.transforms.append((self.reshape_keys, transform_params))
        elif transform_name == 'reshape_peaks':
            self.transforms.append((self.reshape_peaks, transform_params))
        elif transform_name == 'check_signal_shape':
            self.transforms.append((self.check_signal_shape, transform_params))
        elif transform_name == 'add_noise_signal':
            self.transforms.append((self.add_noise_signal, transform_params))
        elif transform_name == 'scale_segment':
            self.transforms.append((self.scale_segment, None))
        elif transform_name == 'scale_multiple_segments':
            self.transforms.append((self.scale_multiple_segments, None))
        elif transform_name == 'pop_keys':
            self.transforms.append((self.pop_keys, transform_params))
        elif transform_name == 'check_nans':
            self.transforms.append((self.check_nans, transform_params))
        elif transform_name == 'print_keys':
            self.transforms.append((self.print_keys, transform_params))
        else:
            raise NotImplementedError

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

    def duplicate_fecg(self, signal_dict, key_to_dupe, *keys):
        for key in keys:
            signal_dict[key] = np.zeros_like(signal_dict[key_to_dupe])

    def perform_trim(self, signal_dict, desired_length, offset, *keys):
        '''trim if necessary to reduce computational load on resampling'''
        trim_length = int(desired_length + offset)
        for key in keys:
            if signal_dict[key].shape[1] < trim_length:
                return

            signal_dict[key] = signal_dict[key][:, offset:trim_length]

    def trim_peaks(self, signal_dict, desired_length, offset):
        trim_length = int(desired_length + offset)
        signal_dict['fecg_peaks'] = signal_dict['fecg_peaks'][(signal_dict['fecg_peaks'] < trim_length) &
                                                              (offset < signal_dict['fecg_peaks'])].astype(int) - offset

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
            signal_dict[mask_key], signal_dict[binary_mask_key] = calc_multi_channel_peak_mask(signal_dict[sig_key])
        else:
            signal_dict[mask_key], signal_dict[binary_mask_key] = calc_multi_channel_peak_mask(signal_dict[sig_key],
                                                                                               actual_peaks=
                                                                                               signal_dict[peak_key],)

    def reshape_keys(self, signal_dict, *keys):
        '''reshape signal into n_segments x segment for individual scaling'''
        for key in keys:
            signal_dict[key] = signal_dict[key].reshape(NUM_WINDOWS, WINDOW_LENGTH)

    def reshape_peaks(self, signal_dict, peak_key, pad_length=PAD_LENGTH):
        peak_locs = np.zeros((NUM_WINDOWS, pad_length))
        for window in range(NUM_WINDOWS):
            borders = (window*WINDOW_LENGTH, (window+1)*WINDOW_LENGTH)
            between = np.logical_and(signal_dict[peak_key] > borders[0], signal_dict[peak_key] < borders[1])
            bordered_peaks = (signal_dict[peak_key][between] - borders[0]) / WINDOW_LENGTH
            pad_amount = pad_length-bordered_peaks.shape[0]
            assert pad_amount >= 0
            padded_peaks = np.pad(bordered_peaks, pad_width=(0, pad_amount), mode='constant')
            peak_locs[window,:] = padded_peaks

        signal_dict[peak_key] = peak_locs

    def check_signal_shape(self, signal_dict, key1, key2):
        assert signal_dict[key1].shape == signal_dict[key2].shape, f'{signal_dict[key1].shape} {signal_dict[key2].shape}'

    def check_nans(self, signal_dict, *keys_to_check):
        for key in keys_to_check:
            assert not np.isnan(signal_dict[key]).any(), f'Signals are nan {signal_dict[key]}'

    def add_noise_signal(self, signal_dict, noise_key, shape_key):
        '''adds a gaussian noise (random) '''
        signal_dict[noise_key] += generate_gaussian_noise_by_shape(shape=signal_dict[shape_key].shape, stdev=NOISE).numpy()

    def scale_segment(self, signal_dict):
        mf_ratio = get_random_mfratio(MF_RATIO, MF_RATIO_STD)

        signal_dict['mf_ratio'] = mf_ratio

        signal_dict['mecg_sig'], signal_dict['fecg_sig'], signal_dict['offset'] = scale_signals(signal_dict['mecg_sig'],
                                                                                                signal_dict['fecg_sig'],
                                                                                                mf_ratio,
                                                                                                signal_dict['noise'])

    def scale_multiple_segments(self, signal_dict):
        '''scales multiple segments'''
        mf_ratio = get_random_mfratio(MF_RATIO, MF_RATIO_STD)
        signal_dict['offset'] = np.zeros_like(signal_dict['mecg_sig'])
        signal_dict['mf_ratio'] = mf_ratio
        for i in range(NUM_WINDOWS):
            signal_dict['mecg_sig'][i, :], signal_dict['fecg_sig'][i, :], signal_dict['offset'][i, :] = scale_signals(
                signal_dict['mecg_sig'][[i], :],
                signal_dict['fecg_sig'][[i], :], mf_ratio,
                signal_dict['noise'][[i], :])

    def pop_keys(self, signal_dict, *keys_to_pop):
        for key in keys_to_pop:
            signal_dict.pop(key)

    def print_keys(self, signal_dict, *keys_to_print):
        for key in keys_to_print:
            print(f'{key} : {signal_dict[key]}')