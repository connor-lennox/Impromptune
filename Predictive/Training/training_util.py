import numpy as np
import torch


def data_from_samples(samples, given=24):
    xs = [
        sample[i:i+given] for sample in samples for i in range(0, len(sample) - given)
    ]

    ys = [
        sample[i+given] for sample in samples for i in range(0, len(sample) - given)
    ]

    return torch.tensor(xs), torch.tensor(ys)


def create_train_test(samples, train_ratio=0.8, given=24):
    train_cutoff = int(len(samples) * train_ratio)
    x_train, y_train = data_from_samples(samples[:train_cutoff], given)
    x_test, y_test = data_from_samples(samples[train_cutoff:], given)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    return x_train, y_train, x_test, y_test


def accuracy(predicted, real):
    predictions = np.argmax(predicted.detach().numpy(), axis=1)
    real_np = real.detach().numpy()
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
