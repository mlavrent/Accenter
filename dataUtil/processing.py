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
    audio_file = AudioSegment.from_wav(filepath)
    audio_file = audio_file[:audio_length * MS]
    ranges = detect_nonsilent(audio_file, min_silence_len=silence_length,
                              silence_thresh=silence_thresh)
    return ranges


def pad_audio_clip(clip, sample_rate):
    seg_target_len = SEG_LEN * sample_rate
    pad_len = int(seg_target_len - len(clip))
    return np.pad(clip, ((0, pad_len), (0, 0)), mode='constant')


def split_audio_clip(clip, sample_rate):
    split = []
    n = math.ceil(len(clip) / (sample_rate * SEG_LEN))
    for s in np.array_split(clip, n):
        split.append(pad_audio_clip(s, sample_rate))
    return split


def merge_audio_clips(data, ranges, sample_rate):
    segmented = []
    for start, end in ranges:
        f_start, f_end = int((start / MS) * sample_rate),\
                         int((end / MS) * sample_rate)
        seg = data[f_start:f_end]
        if len(seg) / sample_rate > SEG_LEN:
            segmented.extend(split_audio_clip(seg, sample_rate))
        else:
            segmented.append(pad_audio_clip(seg, sample_rate))
    return np.asarray(segmented)


def process_audio_file(filepath, label, export=False):
    if filepath.lower().endswith(".wav"):
        filename = ntpath.basename(filepath)
        # Sampling rate, given in Hz, is number of measurements per second
        sample_rate, data = sio.wavfile.read(filepath)
        non_silent_ranges = get_non_silent_ranges(filepath, 120)

        # Merge audio data by non_silent_ranges
        merged = merge_audio_clips(data, non_silent_ranges, sample_rate)
        if export:
            export_segmented_audio_wav(merged, filename, label, sample_rate)
        return merged
    else:
        print("File format not recognized.")
        exit(1)


def process_audio_directory(path, label, export=False):
    wav_files = [f for f in glob.glob(path + "**/*.wav", recursive=False)]
    audio_data = []
    for f in wav_files:
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
