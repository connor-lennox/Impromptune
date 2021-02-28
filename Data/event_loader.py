import os
import random
import pickle

from Data import midi_to_events


maestro_root = r"Data\Datasets\maestro-v3.0.0"
maestro_events_root = r"Data\Datasets\maestro-events"

# 333-Event Datasets
MAESTRO_EVENTS_SMALL = "maestro-events-small.pkl"
MAESTRO_EVENTS_SMALL_DENSE = "maestro-events-small-dense.pkl"

MAESTRO_EVENTS_MEDIUM = "maestro-events-medium.pkl"

MAESTRO_EVENTS_FULL = "maestro-events-full.pkl"

# 240-Event Datasets
MAESTRO_EVENTS_MEDIUM_240 = "maestro-events-medium-240.pkl"
MAESTRO_EVENTS_2017_240 = "maestro-events-2017-240.pkl"
MAESTRO_EVENTS_FULL_240 = "maestro-events-full-240.pkl"


def find_maestro_midis(files_root=maestro_root):
    return [os.path.join(root, name)
            for root, _, files in os.walk(files_root)
            for name in files if name.endswith('.midi')]


def sample_from_maestro(file_root=maestro_root, songs_to_sample=100, samples_per_song=1):
    files = find_maestro_midis(file_root)
    if songs_to_sample is not None:
        files = random.sample(find_maestro_midis(), songs_to_sample)
    samples = [sample for song in files for sample in midi_to_events.read_midi(song, sample_size=samples_per_song)]
    return samples


def pickle_data(filename, data):
    with open(os.path.join(maestro_events_root, filename), 'wb+') as outfile:
        pickle.dump(data, outfile)


def load_dataset(filename):
    with open(os.path.join(maestro_events_root, filename), 'rb+') as infile:
        return pickle.load(infile)


if __name__ == '__main__':
    # Generate randomly sampled dataset
    # test_samples = sample_from_maestro(songs_to_sample=150, samples_per_song=None)
    # pickle_data("maestro-events-medium-240.pkl", test_samples)

    # Generate 2017 dataset
    samples = sample_from_maestro(maestro_root, songs_to_sample=None, samples_per_song=None)
    pickle_data("maestro-events-full-240.pkl", samples)
