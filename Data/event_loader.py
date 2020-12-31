import os
import random
import pickle

from Data import midi_to_events


maestro_root = r"C:\Users\Connor\Documents\Research\Impromptune\Data\Datasets\maestro-v3.0.0"
maestro_events_root = r"C:\Users\Connor\Documents\Research\Impromptune\Data\Datasets\maestro-events"

MAESTRO_EVENTS_SMALL = "maestro-events-small.pkl"


def find_maestro_midis():
    return [os.path.join(root, name)
            for root, _, files in os.walk(maestro_root)
            for name in files if name.endswith('.midi')]


def sample_from_maestro(songs_to_sample=100, samples_per_song=1):
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
    # test_samples = sample_from_maestro()
    # pickle_data("Datasets/maestro-events/maestro-events-small", test_samples)
    print(len(load_dataset(MAESTRO_EVENTS_SMALL)))
