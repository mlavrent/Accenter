from matplotlib import pyplot as plt
import numpy as np
from constants import *
from scipy.io import wavfile


def export_audio_data(filepath, data):
    """
    Exports the given audio data to the provided file path

    :param filepath:    string of directory to export to
    :param data:        n-dimensional Numpy array
    """
    np.save(filepath, data)


def read_audio_data(filepath):
    """
    Reads the processed audio data from the provided file path

    :param filepath:    string of file to import from
    :return:            n-dimensional Numpy array
    """
    return np.load(filepath)


def plot_audio_segment(sample_rate, data, ranges, filename="plot.png"):
    """
    Plots the given audio data based on sample_rate and non-silent ranges on a
    matplotlib plot and exports to a given filename

    :param sample_rate:     frames per second from the .wav audio file
    :param data:            raw audio date Numpy array with shape [MS, 2]
    :param ranges:          non-silent ranges from audio data
    :param filename:        filename to save plot to (defaults to plot.png)
    """
    sample_rate = float(sample_rate)
    times = np.arange(len(data)) / sample_rate
    plt.figure(figsize=(300, 10))
    plt.xlim(times[0], times[-1])
    plt.fill_between(times, data[:, 0], data[:, 1], color='k')
    for start, end in ranges:
        if end / MS < times[-1]:
            plt.axvspan(start / MS, end / MS, alpha=0.2, color='red')
        else:
            break

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude s(t)')
    plt.savefig(filename, dpi=100)
    plt.show()


def export_segmented_audio_wav(audio_clips, filename, label, sample_rate):
    """
    Export the segmented audio data into smaller clips of non-silent speech

    :param audio_clips:     AudioSegment audio files
    :param filename:        base filename to export to
    :param label:           accent speech label
    :param sample_rate:     frames per second from the .wav audio file
    """
    for i, segment in enumerate(audio_clips):
        wavfile.write("./data/processed/{0}/clips/{1}-clip{2}.wav".format(
            label, filename, i), sample_rate, segment)
