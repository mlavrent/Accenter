import numpy as np
import glob
import math
import ntpath
import scipy.io as sio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from ioUtil import *
from constants import *


def get_non_silent_ranges(filepath, audio_length, silence_length=1000,
                          silence_thresh=-48):
    """
    Given a filepath to a .wav file and a target audio length, return all the
    non-silent ranges from the audio sample

    :param filepath:        filepath to the .wav audio file
    :param audio_length:    length in seconds of audio to process
    :param silence_length:  minimum length of a silence to be used for
                                a split
    :param silence_thresh: (in dBFS) anything quieter than this will be
                                considered silence
    :return: 2D array:      array of shape [num_ranges, 2]
    """
    # Load data into AudioSegment object and extract audio_length seconds
    audio_file = AudioSegment.from_wav(filepath)
    audio_file = audio_file[:audio_length * MS]
    # Use given parameters to return non_silent ranges
    ranges = detect_nonsilent(audio_file, min_silence_len=silence_length,
                              silence_thresh=silence_thresh)
    return ranges


def pad_audio_clip(clip, sample_rate):
    """
    Pad an audio slip with silence to SEGMENT_LENGTH

    :param clip:            audio data to pad
    :param sample_rate:     frames per second from the .wav audio file
    :return: Numpy array    array of shape [SEGMENT_LENGTH * sample_rate, 2]
    """
    # Target length for segment is SEGMENT_LENGTH * sample_rate
    seg_target_len = SEGMENT_LENGTH * sample_rate
    pad_len = int(seg_target_len - len(clip))
    # Pad the audio clip on the bottom by pad_len
    return np.pad(clip, ((0, pad_len), (0, 0)), mode='constant')


def split_audio_clip(clip, sample_rate):
    """
    Splits a given audio clip into SEGMENT_LENGTH intervals, padding the rest
    with silence

    :param clip:            The full length audio sample to split
    :param sample_rate:     frames per second from the .wav audio file
    :return: array          array of Numpy arrays of shape
                            [SEGMENT_LENGTH * sample_rate, 2]
    """
    split = []
    # Number of segments in audio clip that can be made with len at most SEG_LEN
    n = math.ceil(len(clip) / (sample_rate * SEGMENT_LENGTH))
    for s in np.array_split(clip, n):
        split.append(pad_audio_clip(s, sample_rate))
    return split


def segment_audio_clips(data, ranges, sample_rate):
    """
    Segments the given audio data by extracting the non-silent intervals given
    the non-silent ranges

    :param data:            Numpy array of shape [num_frames, 2]
    :param ranges:          non-silent ranges from audio data
    :param sample_rate:     frames per second from the .wav audio file
    :return: Numpy array    segmented audio data with shape
                            [num_examples, SEGMENT_LENGTH * sample_rate, 2]
    """
    segmented = []
    for start, end in ranges:
        # Convert start, end (in seconds) to f_start, f_end (in frames)
        f_start, f_end = int((start / MS) * sample_rate),\
                         int((end / MS) * sample_rate)
        # Slice raw audio data to non_silent segment
        seg = data[f_start:f_end]
        if len(seg) / sample_rate > SEGMENT_LENGTH:
            # Extend by splitting audio clips into padded SEG_LEN intervals
            segmented.extend(split_audio_clip(seg, sample_rate))
        else:
            # Pad current audio clip
            segmented.append(pad_audio_clip(seg, sample_rate))
    # Convert to 3D np array
    return np.asarray(segmented)


def process_audio_file(filepath, label, export=False):
    """
    Processes a single audio file given a filepath to the audio data.
    Accepted data forms: .wav

    :param filepath:        string representing filepath to audio data
    :param label:           accent speech label
    :param export:          Boolean value whether to export the processed data
    :return: Numpy array    segmented audio data with shape
                            [num_examples, SEGMENT_LENGTH * sample_rate, 2]
    """
    if filepath.lower().endswith(".wav"):
        filename = ntpath.basename(filepath).split(".wav")[0]

        # Sampling rate, given in Hz, is number of measurements per second
        sample_rate, data = sio.wavfile.read(filepath)
        non_silent_ranges = get_non_silent_ranges(filepath, 120)
        # Segment audio data by non_silent_ranges
        segmented = segment_audio_clips(data, non_silent_ranges, sample_rate)
        if export:
            export_segmented_audio_wav(segmented, filename, label, sample_rate)
            plot_audio_segment(sample_rate, ranges=non_silent_ranges,
                               data=data[:120 * sample_rate], filename=filename)
        return segmented
    else:
        print("File format not recognized.")
        exit(1)


def process_audio_directory(path, label, export=False):
    """
    Process all audio files from a given path to the directory. Combines all
    processed audio data from each file into one 3D Numpy array containing
    all audio data for a given accent group

    :param path:            path to a directory containing audio data
    :param label:           accent speech label
    :param export:          Boolean value whether to export the processed data
    :return: Numpy array    segmented audio data with shape
                            [num_examples, SEGMENT_LENGTH * sample_rate, 2]
    """
    # Read all .wav files in provided directory
    wav_files = [f for f in glob.glob(path + "**/*.wav", recursive=False)]
    audio_data = []
    for f in wav_files:
        # Get parsed audio data for each file
        audio_data.append(process_audio_file(f, label, export=False))
    audio_data = np.concatenate(audio_data)
    if export:
        filename = "./data/processed/{0}/{1}".format(label, ntpath.basename(path))
        export_audio_data(filename, audio_data)
    return audio_data


if __name__ == '__main__':
    a = process_audio_directory("./data/raw/english", "english", True)
    print(a.shape)
    b = read_audio_data("./data/processed/english/english.npy")
    print(b.shape)

    a = process_audio_directory("./data/raw/chinese", "chinese", True)
    print(a.shape)
    b = read_audio_data("./data/processed/chinese/chinese.npy")
    print(b.shape)

    a = process_audio_directory("./data/raw/british", "british", True)
    print(a.shape)
    b = read_audio_data("./data/processed/british/british.npy")
    print(b.shape)
