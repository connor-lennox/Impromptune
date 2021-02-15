import numpy as np
from matplotlib import pyplot as plt
import torch

import Predictive.Models.model_persistence as persistence
import Data.event_loader as event_loader
import Predictive.Training.training_util as training_util


def make_predictions(model, xs):
    return torch.tensor([model(x)[0] for x in xs])


def confusion_matrix(model, xs, ys, num_classes=333):
    y_pred = make_predictions(model, xs)

    matrix = np.zeros((num_classes, num_classes))

    for real, pred in zip(ys, y_pred):
        matrix[real, pred] += 1

    return matrix


if __name__ == '__main__':
    model_to_test = ""
    model = persistence.unpickle_model(model_to_test)

    given_elems = 500
    _, data = training_util.create_train_test(event_loader.load_dataset(event_loader.MAESTRO_EVENTS_MEDIUM), given=given_elems)

    xs = data.xs
    ys = data.ys

    conf_mat = confusion_matrix(model, xs, ys)

    print(conf_mat)
