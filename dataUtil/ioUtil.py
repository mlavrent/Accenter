from matplotlib import pyplot as plt
import numpy as np
from constants import *
from scipy.io import wavfile


def export_audio_data(filepath, data):
    np.save(filepath, data)


def read_audio_data(filepath):
    return np.load(filepath)


def plot_audio_segment(sample_rate, data, overlayed=False, ranges=None,
                       filename="plot.png"):
    sample_rate = float(sample_rate)
    times = np.arange(len(data)) / sample_rate
    plt.figure(figsize=(300, 10))
    plt.xlim(times[0], times[-1])
    plt.fill_between(times, data[:, 0], data[:, 1], color='k')
    if overlayed:
        for start, end in ranges:
            if end // MS < times[-1]:
                plt.axvspan(start / MS, end / MS, alpha=0.2, color='red')
            else:
                break

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude s(t)')
    plt.savefig(filename, dpi=100)
    plt.show()


def export_segmented_audio_wav(audio_clips, filename, label, sample_rate):
    for i, segment in enumerate(audio_clips):
        wavfile.write("./data/processed/{0}/clips/{1}-clip{2}.wav".format(
            label, filename, i), sample_rate, segment)
