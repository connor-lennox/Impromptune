import os
import random

import numpy as np
import pypianoroll


# Notes for myself later:
# 1a. lpd data has 24 samples per "beat", 4 beats per measure.
# 1b. From above, there are 96 samples in a measure.
# 1c. To split into 4 measure chunks, I'll need 384 samples. (Length of training samples and generated data)
# 2a. There are like 21,000 files in the lpd dataset. Each one has over 1 million elements in its piano roll
# 2b. For small scale testing, only a subset of the dataset should be used. Maybe 100 songs?
# 2c. To get a varied spread of music from across the dataset, perhaps 1000 random segments of songs can
#     be sampled as opposed to having multiple segments from a small portion of the songs.


data_root = f"C:/Users/Connor/Documents/Research/Impromptune/Data/Datasets/"

lpd_5_root = data_root + "lpd_5_cleansed/"
lpd_cleaned_root = data_root + "lpd_cleaned/"

beats_per_measure = 4
num_measures = 4
min_notes_per_segment = 10

lowest_pitch = 24
num_pitches = 84


def _track_is_piano(track):
    return track.name == "Piano"


def _roll_not_sparse(roll):
    return (roll.sum() > min_notes_per_segment).any()


def extract_pianoroll(filename):
    roll = pypianoroll.load(filename)
    piano_track = list(filter(_track_is_piano, roll.tracks))[0].pianoroll[:, lowest_pitch:lowest_pitch+num_pitches]
    piano_track = np.clip(piano_track, 0, 1)
    return piano_track, roll.resolution


def sample_segments(filename, max_samples=4):
    roll, beat_resolution = extract_pianoroll(filename)
    seq_len = roll.shape[0]
    sample_length = beat_resolution * beats_per_measure * num_measures
    sample_candidates = [roll[x:x+sample_length, :] for x in range(0, seq_len-sample_length, sample_length)]
    sample_candidates = list(filter(_roll_not_sparse, sample_candidates))
    target_samples = min(len(sample_candidates), max_samples)
    return random.sample(sample_candidates, target_samples)


def load_samples(files, songs_to_sample):
    return np.array([sample for song in random.sample(files, songs_to_sample)
                     for sample in sample_segments(song)])


def sample_data(files_root, sample_ratio=100):
    file_names = [os.path.join(root, name) for root, _, files in os.walk(files_root) for name in files]
    samples = load_samples(file_names, songs_to_sample=len(file_names) // sample_ratio)
    return samples


def sample_lpd5(sample_ratio=100):
    return sample_data(lpd_5_root, sample_ratio)


def sample_lpd_cleansed(sample_ratio=100):
    return sample_data(lpd_cleaned_root, sample_ratio)


if __name__ == '__main__':
    test_segments = sample_lpd5()
    print(test_segments.shape)
