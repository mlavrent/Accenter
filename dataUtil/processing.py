from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

MS = 1000


def plot_audio_segment(sample_rate, data, overlayed=False, ranges=None,
                       filename="plot.png"):
    sample_rate = float(sample_rate)
    times = np.arange(len(data)) / sample_rate
    # TODO: make figsize a function of data dimensions
    plt.figure(figsize=(100, 10))
    plt.xlim(times[0], times[-1])
    plt.fill_between(times, data[:, 0], data[:, 1], color='k')
    # TODO: check range based on data
    if overlayed:
        for start, end in ranges:
            plt.axvspan(start / MS, end / MS, alpha=0.2, color='red')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude s(t)')
    plt.savefig(filename, dpi=100)
    plt.show()


def get_non_silent_ranges(filepath, audio_length, silence_length=1000,
                          silence_thresh=-50):
    audio_file = AudioSegment.from_wav(filepath)
    audio_file = audio_file[:audio_length * MS]
    ranges = detect_nonsilent(audio_file, min_silence_len=silence_length,
                              silence_thresh=silence_thresh)
    return ranges


def process_audio_file(filepath):
    if filepath.lower().endswith(".wav"):
        # Sampling rate, given in Hz, is number of measurements per second
        sample_rate, data = wavfile.read(filepath)
        non_silent_ranges = get_non_silent_ranges(filepath, 100)
        # Multiply second * sample rate gives number of frames in raw WAV data
        plot_audio_segment(sample_rate, data[:100 * sample_rate],
                           overlayed=True,
                           ranges=non_silent_ranges,
                           filename=filepath.split(".wav")[0] + "-segmented"
                                                                "-plot")

    else:
        print("File format not recognized.")
        exit(1)


if __name__ == '__main__':
    process_audio_file("./data/raw/english1.wav")
