import matplotlib.pyplot as plt
import numpy as np
import ioUtil as io

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
    :param mfcc: an array of size (num_segments, num_mfcc)
    :return: None
    """
    specshow(mfcc_data.T, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    plt.show()


def main():
    a = io.read_audio_data("./data/processed/english/english.npy").astype(
        np.float32)
    b = io.read_audio_data("./data/processed/chinese/chinese.npy").astype(
        np.float32)
    c = io.read_audio_data("./data/processed/british/british.npy").astype(
        np.float32)

    segment_times1, sample_frequencies1, spectrogram1 = get_fft(a,
                                                                SAMPLE_RATE,
                                                                testing=False)
    segment_times2, sample_frequencies2, spectrogram2 = get_fft(b,
                                                                SAMPLE_RATE,
                                                                testing=False)
    segment_times3, sample_frequencies3, spectrogram3 = get_fft(c, SAMPLE_RATE,
                                                                testing=False)

    mfcc1 = get_mfcc(a, SAMPLE_RATE, 12, testing=False)
    mfcc2 = get_mfcc(b, SAMPLE_RATE, 12, testing=False)
    mfcc3 = get_mfcc(c, SAMPLE_RATE, 12, testing=False)

    print(a.shape)
    print(b.shape)
    print(c.shape)
    print()
    print(spectrogram1.shape)
    print(spectrogram2.shape)
    print(spectrogram3.shape)
    print()
    print(mfcc1.shape)
    print(mfcc2.shape)
    print(mfcc3.shape)

    plot_spectrogram(segment_times1,
                     sample_frequencies1,
                     spectrogram1[0])
    plot_mfcc(mfcc1[0])


if __name__ == '__main__':
    main()
