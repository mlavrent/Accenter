import glob
import math

import scipy.io as sio
import numpy as np
import ioUtil as io

from tqdm import trange
from pathlib import Path
from librosa import load
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from constants import *


def flatten_audio_channels(data):
    """
    Flatten the audio data with multiple channels by averaging the data from
    multiple channels

    :param data:            processed audio data of shape
                            [num_examples, SEGMENT_LENGTH * sample_rate, 2]
    :return: Numpy array    flattened audio data of shape
                            [num_examples, SEGMENT_LENGTH * sample_rate * 2]
    """
    return np.mean(data, axis=2)


def get_non_silent_ranges(filepath, audio_length, silence_length,
                          silence_thresh):
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
    ranges = []
    for i in range(10):
        ranges = detect_nonsilent(audio_file,
                                  min_silence_len=int(silence_length),
                                  silence_thresh=silence_thresh)
        if len(ranges) > 0:
            break
        else:
            silence_length /= 2
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
        f_start, f_end = int((start / MS) * sample_rate), \
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


def process_audio_file(filepath, label, silence_length, silence_thresh,
                       testing=False):
    """
    Processes a single audio file given a filepath to the audio data.
    Accepted data forms: .wav

    :param filepath:        string representing filepath to audio data
    :param label:           accent speech label
    :param silence_length:  minimum length of a silence to be used for
                                a split
    :param silence_thresh:  (in dBFS) anything quieter than this will be
                                considered silence
    :param testing:         Boolean value whether to run as testing
    :return: Numpy array    segmented audio data with shape
                            [num_examples, SEGMENT_LENGTH * sample_rate, 2]
    """
    path = Path(filepath)
    if path.suffix == '.wav':
        # Sampling rate, given in Hz, is number of measurements per second
        sample_rate, data = sio.wavfile.read(filepath)
        if sample_rate != SAMPLE_RATE:
            print("ERROR: Invalid sample rate detected. Resampling file")
            resample_wav_file(filepath, testing=testing)
            _, data = sio.wavfile.read(filepath)

        audio_length = NUM_TESTING_CLIPS if testing else len(data)
        non_silent_ranges = get_non_silent_ranges(filepath, audio_length,
                                                  silence_length,
                                                  silence_thresh)
        # Segment audio data by non_silent_ranges
        segmented = segment_audio_clips(data, non_silent_ranges, SAMPLE_RATE)
        # if testing:
        io.export_segmented_audio_wav(segmented, path.stem, label,
                                      SAMPLE_RATE)
        return segmented
    else:
        print("File format not recognized.")
        exit(1)


def process_accent_group(path, label, testing=False, silence_length=1000,
                         silence_thresh=-62):
    """
    Process all audio files from a given path to the directory. Combines all
    processed audio data from each file into one 2D Numpy array containing
    all audio data for a given accent group

    :param path:            path to a directory containing audio data
    :param label:           accent speech label
    :param testing:         Boolean value whether to run as testing
    :param silence_length:  minimum length of a silence to be used for
                                a split
    :param silence_thresh:  (in dBFS) anything quieter than this will be
                                considered silence
    :return: Numpy array    segmented audio data with shape
                            [num_examples, SEGMENT_LENGTH * sample_rate]
    """
    # Read all .wav files in provided directory
    wav_files = [f for f in glob.glob(path + "**/*.wav", recursive=False)]
    audio_data = []
    for i in trange(len(wav_files), desc=f"{label.capitalize()}"):
        f = wav_files[i]
        # Get parsed audio data for each file
        processed = process_audio_file(f, label, silence_length, silence_thresh,
                                       testing=testing)
        if len(processed) > 0:
            audio_data.append(processed)
    # Make sure audio data exists
    if len(audio_data) > 0:
        audio_data = flatten_audio_channels(np.concatenate(audio_data))
        # Always export the audio data to a .npy file
        filename = f"./data/processed/{label}/{Path(path).stem}"
        io.export_audio_data(filename, audio_data)
    return audio_data


def process_audio_directory(path, testing=False, silence_length=1000,
                            silence_thresh=-62):
    """
    Process an entire file directory of audio files where each subdirectory
    contains audio data for a given accent class

    :param path:            path to a directory containing audio data
    :param testing:         Boolean value whether to run as testing
    :param silence_length:  minimum length of a silence to be used for
                                a split
    :param silence_thresh:  (in dBFS) anything quieter than this will be
                                considered silence
    :return: dict           dictionary of key values pairs of the form
                    {accent: [num_examples, SEGMENT_LENGTH * sample_rate]}

    """
    classes = glob.glob(path + "/*")
    audio_data = {}
    for c in classes:
        p = Path(c)
        audio_data[p.stem] = process_accent_group(
            c, p.stem, testing=testing, silence_length=silence_length,
            silence_thresh=silence_thresh)
    return audio_data


def resample_wav_file(filepath, testing=False):
    """
    Resamples a wav file to the standard sampling rate of 22050. Renames the
    original file with a '_ORIGINAL' filename and write the newly generated
    .wav file to the original filepath.

    :param filepath:        path to a wav file of audio to be resampled
    :param testing:         Boolean value whether to run as testing
    :return: None
    """
    duration = NUM_TESTING_CLIPS if testing else None
    data, _ = load(filepath, mono=False, duration=duration)
    p = Path(filepath)
    p.rename(Path(p.parent, f"{p.stem}_ORIGINAL{p.suffix}"))
    sio.wavfile.write(filepath, SAMPLE_RATE, data.T)


def main():
    # resample_wav_file("./data/raw/british/british1.wav", testing=False)
    a = process_audio_directory("./data/raw", testing=False)

    for k, v, in a.items():
        assert(v.shape == io.read_audio_data(
            f"./data/processed/{k}/{k}.npy").shape)
        print(f"{k.capitalize()}: {v.shape}")


if __name__ == '__main__':
    main()
