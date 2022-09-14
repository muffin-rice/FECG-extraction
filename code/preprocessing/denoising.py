import numpy as np
import pywt
from scipy.signal import firwin2, filtfilt, savgol_filter
import math


def fir_filt(window, fs, numtaps=31):
    """
    Remove the baseline ("flattening the segment")
    and powerline noise of segment.
    window ~ signal segment - float array
    """

    if len(window) > 93:
        # Minimum length for filter is 94 to satisfy
        # window.shape[-1] > padlen = 3 * numtaps
        gain = [0, 1, 1, 0, 0]
        freq = [0, 1, 45, 55, fs / 2]

        b = firwin2(numtaps, freq, gain, fs=fs, window='hamming', antisymmetric=True)
        window = filtfilt(b, 1, window)

    return window


def wavelet_padding(signal):
    size = len(signal)
    exp = math.ceil(math.log2(size))
    pad = int(math.pow(2, exp) - size)
    signal = np.concatenate((signal, signal[:size - pad - 1:-1]), axis=-1)

    return signal


def thresholding(dcmp, n):
    sigma = np.mean(np.absolute(dcmp - np.mean(dcmp))) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log10(n))

    return thresh


def wav_filt(window, thresh_sum=0, thresh_cnt=0, enable_th=True, fs=1000, wavelet='coif5', lvl=4):
    original_length = len(window)
    window = wavelet_padding(window)

    great_wave = pywt.swt(window, wavelet, level=lvl)
    great_wave = np.array(great_wave)

    thv = 0

    if enable_th:
        thv = thresholding(great_wave[0, 1, :], len(
            window))  # Universal Threshold
        thresh_cnt += 1

    thresh_sum = thresh_sum + thv  # Store threshold
    great_wave[:, 1, :] = (
        pywt.threshold(great_wave[:, 1, :], thresh_sum / thresh_cnt, 'soft')
    )
    great_wave[lvl - 1, 0, :] = 0
    great_wave[lvl - 1, 1, :] = 0

    window = pywt.iswt(great_wave, wavelet)

    window = window[0:original_length]

    return window, thresh_sum, thresh_cnt


def detect_interp_outlier(window, r_avg, fs=1000):
    """
    Detect, remove, and interpolate Outliers
    ****Inputs****
    window - Signal Segment ~ float array
    r_avg  - Average R-Peak height ~ float
    fs     - Sampling Frequency ~ int
    ****Outputs****
    window     - outlier removed signal segment ~ float array
    True/False - outlier existence in signal
    """

    # Handle First window
    if r_avg == 0:
        return window, True

    # Calculate outlier threshold and find region of interest
    outlier = np.where(abs(window) > 3 * r_avg)[0]

    # Skip extra steps if no outlier
    if len(outlier) == 0:
        return window, True

    # Calculate Region of Interest
    roi = (outlier[:, None] +
           np.arange(int(-fs / 5), int(fs / 5) + 1, 1)).astype(int)
    roi[roi < 0] = 0
    roi[roi > len(window) - 1] = len(window) - 1

    # Narrow outlier to the highest point
    outlier = np.where(
        np.amax(np.abs(window[roi]), 1)[:, None] == np.abs(window))[1]
    outlier = np.unique(outlier)

    # Set region around and at outlier point to NaN
    outlier = np.array(outlier)[:, None] + \
              np.arange(int(-fs / 3), int(fs / 3 + 1), 1)
    outlier[outlier < 0] = 0
    outlier[outlier > len(window) - 1] = len(window) - 1
    window[outlier] = np.nan

    # interpolate
    window = interp_nan(window)

    return window, False


def interp_nan(window):
    """
    Interpolate NaNs after outlier removal.
    window  : 3 sec segment ~ Array float
    """
    window = np.array(window)
    isnan = np.isnan(window)
    nans = np.where(isnan)[0]

    # No NaNs; No interpolation
    if np.size(nans) == 0:
        return window

    # If the whole signal is NaN set signal to zero
    elif len(nans) == len(window):
        window[:] = 0

    # Fill NaNs using linear interpolation (NaNs on edge are copies of outermost float)
    else:
        ok = ~isnan
        xp = ok.ravel().nonzero()[0]
        fp = window[ok]
        x = isnan.ravel().nonzero()[0]
        window[isnan] = np.interp(x, xp, fp)

    return window

def preprocess(window: np.ndarray, sampling_rate, numtaps : int = 0, winlen : int = 0, polyorder : int = 0, wavelet : str = '', lvl : int = 0) -> np.ndarray:
    if window.size == 0:
        return window, window
    
    if numtaps:
        window = fir_filt(window, sampling_rate, numtaps=numtaps)
    light = window
    
    if winlen and polyorder:
        window = savgol_filter(window, winlen, polyorder)
        
    if wavelet and lvl:
        decomp, _, _ = wav_filt(window, wavelet=wavelet, fs=sampling_rate, lvl=lvl)
        window = decomp[0]
    
    return window, light