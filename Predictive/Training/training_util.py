import numpy as np
import torch
from torch.utils.data import Dataset


class EventsDataset(Dataset):
    # A dataset of events-style data
    def __init__(self, xs, ys):
        # xs is a tensor of shape (sequences, event)
        self.xs = xs
        # ys is a tensor of shape (sequences), representing the next event for each x in xs
        self.ys = ys

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]

    def __len__(self):
        return self.xs.shape[0]


def data_from_samples(samples, given=24):
    xs = [
        sample[i:i+given] for sample in samples for i in range(0, len(sample) - given)
    ]

    ys = [
        sample[i+given] for sample in samples for i in range(0, len(sample) - given)
    ]

    return EventsDataset(torch.tensor(xs), torch.tensor(ys))


def create_train_test(samples, train_ratio=0.8, given=24):
    train_cutoff = int(len(samples) * train_ratio)
    train_dataset = data_from_samples(samples[:train_cutoff], given)
    test_dataset = data_from_samples(samples[train_cutoff:], given)

    return train_dataset, test_dataset


def create_weights_for_dataset(dataset, num_classes=240):
    # Weight is inversely proportional to commonality, in an attempt to normalize
    # the ratio of samples in the dataset.
    labels = dataset.ys
    events, counts = torch.unique(labels, return_counts=True)
    num_samples = torch.sum(counts)
    weight_per_class = [0.] * num_classes
    for i in range(len(events)):
        weight_per_class[events[i]] = num_samples / counts[i] if counts[i] != 0 else 0
    weight_per_sample = torch.tensor([weight_per_class[y] for y in labels])
    return weight_per_sample


def accuracy(predicted, real):
    predictions = np.argmax(predicted.detach().cpu().numpy(), axis=1)
    real_np = real.detach().cpu().numpy()
    return np.mean(predictions == real_np)


def progress_string(done, total, bar_length=16, include_count=True):
    num_filled = int(done / total * bar_length)
    num_unfilled = bar_length - num_filled
    bar = "[" + "=" * num_filled
    if num_unfilled != 0:
        bar += '>'
    bar += " " * (num_unfilled-1) + "]"
    if include_count:
        bar += f" [{done}/{total}]"
    return bar


if __name__ == '__main__':
    test_data = [[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10]]
    print(data_from_samples(test_data, given=3))
