import glob

import matplotlib.pyplot as plt
import numpy as np

from tqdm import trange
from pathlib import Path
from librosa.display import specshow
from librosa.feature import mfcc
from scipy.signal import spectrogram as spectro
from dataUtil.processing import process_audio_file, flatten_audio_channels

from dataUtil.constants import *
import dataUtil.ioUtil as io


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
    return mfcc(y=signal, sr=sampling_frequency, n_mfcc=num_mfcc).T[:, 1:]


def get_mfcc(signals, sampling_frequency, num_fcc, label, testing=False):
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
    for i in trange(len(signals), desc=f"{label.capitalize()} MFCC"):
        signal = signals[i]
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
        print(npy_root)
        data = io.read_audio_data(npy_root + ".npy").astype(np.float32)
        print("Running FFT")
        _, _, spectrogram = get_fft(data, SAMPLE_RATE, testing=testing)
        print("Running MFCC")
        mfccs = get_mfcc(data, SAMPLE_RATE, NUM_MFCC, p.stem, testing=testing)

        # Get end index for train using test_fraction
        end_train_spectro = int(len(spectrogram) * (1 - TEST_FRACTION))
        end_train_mfcc = int(len(mfccs) * (1 - TEST_FRACTION))

        train_spectro, test_spectro = spectrogram[:end_train_spectro], \
                                      spectrogram[end_train_spectro:]

        train_mfcc, test_mfcc = mfccs[:end_train_mfcc], mfccs[end_train_mfcc:]

        io.export_audio_data(npy_root + "-spectrogram-train.npy", train_spectro)
        io.export_audio_data(npy_root + "-spectrogram-test.npy", test_spectro)
        io.export_audio_data(npy_root + "-mfcc-train.npy", train_mfcc)
        io.export_audio_data(npy_root + "-mfcc-test.npy", test_mfcc)

        audio_data[p.stem] = (spectrogram, mfccs)
    return audio_data


def segment_and_extract(filepath):
    processed = process_audio_file(filepath, None, silence_length=1000,
                                   silence_thresh=-62)
    if len(processed) == 0:
        return None, None
    data = flatten_audio_channels(processed)
    _, _, spectrogram = get_fft(data, SAMPLE_RATE)
    mfccs = get_mfcc(data, SAMPLE_RATE, NUM_MFCC, "NA")
    return spectrogram, mfccs


def main():
    a = extract_audio_directory("./data/processed", testing=False)
    f = open("results.txt", 'w')
    for k, v, in a.items():
        f.write(f"{k.capitalize()}\n")
        f.write(f"{v.shape}\n")

    # for k, v, in a.items():
    #     print(k.capitalize())
    #     print(v[0].shape, v[1].shape)
    # print(io.read_audio_data("./data/processed/spanish/spanish.npy").shape)

if __name__ == '__main__':
    main()
