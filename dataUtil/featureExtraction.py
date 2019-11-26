import glob

import matplotlib.pyplot as plt
import numpy as np
import ioUtil as io

from pathlib import Path
from librosa.display import specshow
from librosa.feature import mfcc
from scipy.signal import spectrogram as spectro

from constants import *


def get_fft(signal, sampling_frequency, testing=False):
    """
    This function takes in an array, either of size (num_frames,) or
    (num_batches, num_frames) and returns a tuple of times, frequencies, and a
    spectrogram.

    :param signal: the signal to get the spectrogram from
    :param sampling_frequency: the sampling frequency of the signal
    :param testing: whether or not to run as testing
    :return: a tuple of segment times (shape: [num_batches,] num_segments),
             sample frequencies (shape: [num_batches,] num_frequencies),
             spectrogram (shape: [num_batches,] num_segments, num_frequencies)
    """
    if testing:
        signal = signal[:NUM_TESTING_CLIPS]
    frequencies, times, spectrogram = spectro(signal, fs=sampling_frequency,
                                              nfft=2048)
    # Only return important frequencies
    return times, frequencies[:NUM_FREQUENCIES], \
           np.swapaxes(spectrogram, 1, 2)[:, :, :NUM_FREQUENCIES]


def _get_single_mfcc(signal, sampling_frequency, num_mfcc):
    """
    This function takes in an array of size (num_frames,) and returns the mfccs
    of size (num_segments, num_mfcc)

    :param signal: the signal to get the mfccs from
    :param sampling_frequency: the sampling frequency of the signal
    :param num_mfcc: the number of mfccs to extract
    :return: an array of size (num_segments, num_mfcc)
    """
    return mfcc(y=signal, sr=sampling_frequency, n_mfcc=num_mfcc).T


def get_mfcc(signals, sampling_frequency, num_fcc, testing=False):
    """
    This function takes in a batch of signals of size (num_batches, num_frames,)
    and returns the mfccs of size (num_batches, num_segments, num_mfcc)

    :param signals: the signal to get the mfccs from
    :param sampling_frequency: the sampling frequency of the signal
    :param num_fcc: the number of mfccs to extract
    :param testing: whether or not to run as testing
    :return: an array of size (num_batches, num_segments, num_mfcc)
    """
    if testing:
        signals = signals[:NUM_TESTING_CLIPS]
    mfccs = []
    for signal in signals:
        mfccs.append(_get_single_mfcc(signal, sampling_frequency, num_fcc))
    return np.array(mfccs)


def plot_spectrogram(times, frequencies, spectrogram):
    """
    Plot the spectrogram.

    :param times: an array of size (num_segments,)
    :param frequencies: an array of size (num_frequencies,)
    :param spectrogram: a spectrogram of size (num_segments, num_frequencies)
    :return: None
    """
    plt.pcolormesh(times, frequencies, spectrogram.T)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [Sec]")
    plt.show()


def plot_mfcc(mfcc_data):
    """
    Plot the mfccs.

    :param mfcc_data: an array of size (num_segments, num_mfcc)
    :return: None
    """
    specshow(mfcc_data.T, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    plt.show()


def extract_audio_directory(path, testing=False):
    """
    Extract features for an entire file directory of audio data where each
    subdirectory contains a single .npy file of processed data

    :param path:            path to a directory containing audio data
    :param testing:         Boolean value whether to run as testing
    :return: dict           dictionary of key values pairs of the form
            {accent: [num_examples, num_features, SEGMENT_LENGTH * sample_rate]}

    """
    classes = glob.glob(path + "/*")
    audio_data = {}
    for c in classes:
        p = Path(c)
        npy_root = f"{c}/{p.stem}"
        data = io.read_audio_data(npy_root + ".npy").astype(np.float32)
        _, _, spectrogram = get_fft(data, SAMPLE_RATE, testing=testing)
        mfccs = get_mfcc(data, SAMPLE_RATE, NUM_FFC, testing=testing)

        io.export_audio_data(npy_root + "-spectrogram.npy", spectrogram)
        io.export_audio_data(npy_root + "-mfcc.npy", mfccs)

        audio_data[p.stem] = (spectrogram, mfccs)
    return audio_data


def main():
    a = extract_audio_directory("./data/processed", testing=True)

    for k, v, in a.items():
        print(k.capitalize())
        print(v[0].shape, v[1].shape)


if __name__ == '__main__':
    main()
