import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader

import Predictive.Models.model_persistence as persistence
import Data.event_loader as event_loader
import Predictive.Training.training_util as training_util


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_predictions(model, data):
    model.eval()
    results = []
    loader = DataLoader(data, num_workers=2, pin_memory=True)
    num_xs = len(data)
    for i, x in enumerate(loader):
        print("\r" + training_util.progress_string(i+1, num_xs), end="")
        sample = x[0].to(DEVICE)
        res = model(sample)[0]
        results.append(torch.argmax(res).item())
    print()
    return torch.tensor(results)


def confusion_matrix(y_pred, y_star, num_classes=333):
    matrix = np.zeros((num_classes, num_classes))

    for real, pred in zip(y_star, y_pred):
        matrix[real, pred] += 1

    return matrix


def simple_accuracy(y_pred, y_star):
    yp = y_pred.detach().numpy()
    ys = y_star.detach().numpy()

    diff = np.equal(yp, ys)
    return np.count_nonzero(diff) / len(diff)


def f1_score(y_pred, y_star):
    matrix = confusion_matrix(y_pred, y_star)

    true_positives = np.diag(matrix)
    false_positives = np.array([np.sum([matrix[i, j] for i in range(333) if i != j]) for j in range(333)])
    false_negatives = np.array([np.sum([matrix[i, j] for j in range(333) if i != j]) for i in range(333)])

    f1 = [true_positives[i] / (true_positives[i] + .5*(false_negatives[i] + false_positives[i]))
          for i in range(len(true_positives))
          if true_positives[i] + .5*(false_negatives[i] + false_positives[i]) != 0]
    return np.mean(f1)


if __name__ == '__main__':
    model_to_test = "onehot-localattn-relu-pred-k256-v512.model"
    model = persistence.unpickle_model(model_to_test).to(DEVICE)

    given_elems = 500
    _, test_dataset = training_util.create_train_test(event_loader.load_dataset(event_loader.MAESTRO_EVENTS_MEDIUM), given=given_elems)

    predictions = make_predictions(model, test_dataset)

    conf_mat = confusion_matrix(predictions, test_dataset.ys)
    acc = simple_accuracy(predictions, test_dataset.ys)
    f1 = f1_score(predictions, test_dataset.ys)

    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")

    plt.matshow(conf_mat)
    plt.show()
