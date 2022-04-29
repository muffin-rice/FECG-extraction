from copy import deepcopy as copy
import numpy as np
import torch
from ecgdetectors import Detectors
from math import pi, sqrt, exp
from scipy.signal import savgol_filter, filtfilt, firwin, firwin2

DETECTOR = Detectors(125).hamilton_detector


def filt(signal: np.ndarray, numtaps=31, sampling_rate=125):
    '''removes the baseline + power line noise'''

    Fs = sampling_rate

    gain = [0, 1, 1, 0, 0]
    freq = [0, 1, 45, 55, Fs / 2]

    if len(signal.shape) == 1:
        b = firwin2(numtaps, freq, gain, fs=Fs, window='hamming', antisymmetric=True)
        signal = filtfilt(b, 1, signal)

        return signal

    for i in range(signal.shape[0]):
        b = firwin2(numtaps, freq, gain, fs=Fs, window='hamming', antisymmetric=True)
        window = filtfilt(b, 1, signal[i, :])
        signal[i, :] = window

    return signal


def normalize(aecg : np.array):  # returns offset and scale factor for normalization
    # aecg is a n x length array
    offset_unscaled_repeat = np.repeat(aecg.min(axis=-1)[:, np.newaxis], aecg.shape[1], axis=1)
    aecg_positive = aecg - offset_unscaled_repeat
    scale_factor = aecg_positive.max(axis=-1)
    scale_factor_repeat = np.repeat(scale_factor[:, np.newaxis], aecg.shape[1], axis=1)

    offset_scaled = offset_unscaled_repeat / scale_factor_repeat

    return offset_scaled, scale_factor_repeat


def scale_signals(mecg_sig, fecg_sig):
    mecg, fecg = copy(mecg_sig), copy(fecg_sig)
    aecg = mecg + fecg
    offset, scale = normalize(aecg)
    return mecg / scale, fecg / scale, offset


def stft_batch(sig: np.array):
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


def invert_stft_batch(stft_sig : torch.Tensor): #batch_size x 34 x 469

    if len(stft_sig.shape) == 4:
        x = []
        for isig in range(stft_sig.shape[0]): #iterate through batch
            stft_sig1 = stft_sig[isig, ...]
            complex_stft = torch.complex(stft_sig1[:,:17,:], stft_sig1[:,17:,:])
            sig = torch.istft(complex_stft, n_fft=32, hop_length=1, onesided=True, return_complex=False, length=500, center=False)

            x.append(sig.unsqueeze(0))


        return torch.cat(x)

    else:

        stft_sig1 = stft_sig
        complex_stft = torch.complex(stft_sig1[:,:17,:], stft_sig1[:,17:,:])

        return torch.istft(complex_stft, n_fft=32, hop_length=1, onesided=True, return_complex=False, length=500, center=False)


def return_peaks(sig : np.array):
    # returns the indices of the R peaks of a 1D array (depends on the algorithm)
    return DETECTOR(sig)

def calc_peak_stats(orig_peaks : [int], pred_peaks : [int], window_size : int = 7):
    # returns the indices of the R peaks of a 1D array (depends on the algorithm)
    ind = 0
    fp = 0
    tp = 0
    fn = 0

    for opeak in orig_peaks:
        omin = opeak - window_size
        omax = opeak + window_size
        while ind < len(pred_peaks):
            if omin <= pred_peaks[ind] <= omax:
                ind += 1
                tp += 1
                break
            elif pred_peaks[ind] > omin:
                fn += 1
                break
            ind += 1
            fp += 1

    precision = tp / (tp + fp) if tp + fp else 1
    recall = tp / len(orig_peaks) if orig_peaks else int(bool(tp))

    return precision, recall, 2 * (precision * recall / (precision + recall) if precision + recall else 0)

def gauss_kernel(n=5,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return np.array([1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r])