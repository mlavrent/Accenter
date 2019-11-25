import numpy as np
from scipy.signal import spectrogram as spectro
from dataUtil.processing import read_audio_data
from librosa.feature import mfcc
from librosa.display import specshow
import matplotlib.pyplot as plt


def get_fft(signal, sampling_frequency):
    '''
    This function takes in an array, either of size (num_frames,) or (num_batches, num_frames)
        and returns a tuple of times, frequencies, and a spectrogram.
    :param signal: the signal to get the spectrogram from
    :param sampling_frequency: the sampling frequency of the signal
    :param window_size: the number of samples per time segment (column) of the spectrogram
    :return: a tuple of segment times (shape: [num_batches,] num_segments),
                        sample frequncies (shape: [num_baches,] num_frequencies),
                        spectrogram (shape: [num_batches,] num_segments, num_frequencies)
    '''
    frequencies, times, spectrogram = spectro(signal, fs=sampling_frequency, nfft=2048)
    # Only return important frequencies
    return times, frequencies[:70], np.swapaxes(spectrogram, 1, 2)[:, :, :70]


def _get_single_mfcc(signal, sampling_frequency, num_mfcc):
    '''
    This function takes in an array of size (num_frames,) and returns the mfccs of size (num_segments, num_mfcc)
    :param signal: the signal to get the mfccs from
    :param sampling_frequency: the sampling frequency of the signal
    :param num_mfcc: the number of mfccs to extract
    :return: an array of size (num_segments, num_mfcc)
    '''
    return mfcc(y=signal, sr=sampling_frequency, n_mfcc=num_mfcc).T

def get_mfcc(signals, sampling_frequency, num_fcc):
    '''
    This function takes in a batch of signals of size (num_batches, num_frames,)
        and returns the mfccs of size (num_batches, num_segments, num_mfcc)
    :param signals: the signal to get the mfccs from
    :param sampling_frequency: the sampling frequency of the signal
    :param num_fcc: the number of mfccs to extract
    :return: an array of size (num_batches, num_segments, num_mfcc)
    '''
    mfccs = []
    for signal in signals:
        mfccs.append(_get_single_mfcc(signal, sampling_frequency, num_fcc))
    return np.array(mfccs)

def plot_spectrogram(times, frequencies, spectrogram):
    '''
    Plot the spectrogram.
    :param times: an array of size (num_segments,)
    :param frequencies: an array of size (num_frequencies,)
    :param spectrogram: a spectrogram of size (num_segments, num_frequencies)
    :return: None
    '''
    plt.pcolormesh(times, frequencies, spectrogram.T)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [Sec]")
    plt.show()


def plot_mfcc(mfcc):
    '''
    Plot the mfccs.
    :param mfcc: an array of size (num_segments, num_mfcc)
    :return: None
    '''
    specshow(mfcc.T, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    plt.show()


def main():
    a = read_audio_data("../data/processed/english/english.npy").astype(np.float32)
    b = read_audio_data("../data/processed/chinese/chinese.npy").astype(np.float32)
    c = read_audio_data("../data/processed/british/british.npy").astype(np.float32)

    english_chinese_sr = 22050
    british_sr = 44100
    #sample_frequencies1, segment_times1, spectrogram1 = get_fft(a, english_chinese_sr)
    segment_times2, sample_frequencies2, spectrogram2 = get_fft(b, english_chinese_sr)
    segment_times3, sample_frequencies3, spectrogram3 = get_fft(c, british_sr)

    # mfcc1 = get_mfcc(a, english_chinese_sr, 12)
    mfcc2 = get_mfcc(b, english_chinese_sr, 12)
    mfcc3 = get_mfcc(c, british_sr, 12)

    print(a.shape)
    print(b.shape)
    print(c.shape)
    print()
    #print(spectrogram1.shape)
    print(spectrogram2.shape)
    print(spectrogram3.shape)
    print()
    # print(mfcc1.shape)
    print(mfcc2.shape)
    print(mfcc3.shape)

    plot_spectrogram(segment_times2, sample_frequencies2, spectrogram2[0])
    plot_mfcc(mfcc2[0])


if __name__ == '__main__':
    #main()
    pass


#https://dsp.stackexchange.com/questions/46516/scipy-signal-spectrogram
#https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
#https://python-speech-features.readthedocs.io/en/latest/#