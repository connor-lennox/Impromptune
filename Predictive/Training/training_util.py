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
    return x_train, y_train, x_test, y_test


def accuracy(predicted, real):
    predictions = np.argmax(predicted.detach().numpy(), axis=1)
    real_np = real.detach().numpy()
    return np.mean(predictions == real_np)


if __name__ == '__main__':
    test_data = [[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10]]
    print(data_from_samples(test_data, given=3))
