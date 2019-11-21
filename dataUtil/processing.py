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
                          silence_thresh=-52):
    # Load data into AudioSegment object and extract audio_length seconds
    audio_file = AudioSegment.from_wav(filepath)
    audio_file = audio_file[:audio_length * MS]
    # Use given parameters to return non_silent ranges
    ranges = detect_nonsilent(audio_file, min_silence_len=silence_length,
                              silence_thresh=silence_thresh)
    return ranges


def pad_audio_clip(clip, sample_rate):
    # Target length for segment is SEG_LEN * sample_rate
    seg_target_len = SEG_LEN * sample_rate
    pad_len = int(seg_target_len - len(clip))
    # Pad the audio clip on the bottom by pad_len
    return np.pad(clip, ((0, pad_len), (0, 0)), mode='constant')


def split_audio_clip(clip, sample_rate):
    split = []
    # Number of segments in audio clip that can be made with len at most SEG_LEN
    n = math.ceil(len(clip) / (sample_rate * SEG_LEN))
    for s in np.array_split(clip, n):
        split.append(pad_audio_clip(s, sample_rate))
    return split


def merge_audio_clips(data, ranges, sample_rate):
    segmented = []
    for start, end in ranges:
        # Convert start, end (in seconds) to f_start, f_end (in frames)
        f_start, f_end = int((start / MS) * sample_rate),\
                         int((end / MS) * sample_rate)
        # Slice raw audio data to non_silent segment
        seg = data[f_start:f_end]
        if len(seg) / sample_rate > SEG_LEN:
            # Extend by splitting audio clips into padded SEG_LEN intervals
            segmented.extend(split_audio_clip(seg, sample_rate))
        else:
            # Pad current audio clip
            segmented.append(pad_audio_clip(seg, sample_rate))
    # Convert to 3D np array
    return np.asarray(segmented)


def process_audio_file(filepath, label, export=False):
    if filepath.lower().endswith(".wav"):
        filename = ntpath.basename(filepath)
        # Sampling rate, given in Hz, is number of measurements per second
        sample_rate, data = sio.wavfile.read(filepath)
        non_silent_ranges = get_non_silent_ranges(filepath, len(data))
        # Merge audio data by non_silent_ranges
        merged = merge_audio_clips(data, non_silent_ranges, sample_rate)
        if export:
            export_segmented_audio_wav(merged, filename, label, sample_rate)
        return merged
    else:
        print("File format not recognized.")
        exit(1)


def process_audio_directory(path, label, export=False):
    # Read all .wav files in provided directory
    wav_files = [f for f in glob.glob(path + "**/*.wav", recursive=False)]
    audio_data = []
    for f in wav_files:
        # Get parsed audio data for each file
        audio_data.append(process_audio_file(f, label, export))
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
