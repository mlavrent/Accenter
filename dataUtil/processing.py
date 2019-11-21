import numpy as np
import math
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


def process_audio_file(filepath, savefile):
    if filepath.lower().endswith(".wav"):
        filename = filepath.split(".wav")[0].split("/")[-1]
        # Sampling rate, given in Hz, is number of measurements per second
        sample_rate, data = sio.wavfile.read(filepath)
        non_silent_ranges = get_non_silent_ranges(filepath, 240)
        # Multiply second * sample rate to get number of frames in raw WAV data
        # plot_audio_segment(sample_rate, data[:130 * sample_rate],
        #                    overlayed=True,
        #                    ranges=non_silent_ranges,
        #                    filename=filepath.split(".wav")[0] + "-segmented"
        #                                                         "-plot2")
        merged = merge_audio_clips(data, non_silent_ranges, sample_rate)
        export_segmented_audio_wav(merged[:3], filename, sample_rate)
        export_audio_data(savefile, merged)

    else:
        print("File format not recognized.")
        exit(1)


if __name__ == '__main__':
    process_audio_file("./data/raw/english1.wav", "./data/processed/test1.npy")
