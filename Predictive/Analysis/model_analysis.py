import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader

import Predictive.Models.model_persistence as persistence
import Data.event_loader as event_loader
import Predictive.Training.training_util as training_util


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVENTS = 240


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


def confusion_matrix(y_pred, y_star, num_classes=NUM_EVENTS):
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
    false_positives = np.array([np.sum([matrix[i, j] for i in range(NUM_EVENTS) if i != j]) for j in range(NUM_EVENTS)])
    false_negatives = np.array([np.sum([matrix[i, j] for j in range(NUM_EVENTS) if i != j]) for i in range(NUM_EVENTS)])

    f1 = [true_positives[i] / (true_positives[i] + .5*(false_negatives[i] + false_positives[i]))
          for i in range(len(true_positives))
          if true_positives[i] + .5*(false_negatives[i] + false_positives[i]) != 0]
    return np.mean(f1)


def generate_probs(model, inputs, length):
    """Modified version of generation code that is always deterministic and keeps track
    of the probabilities for given elements (does not even return generated sequences)

    Used for mean reciprocal  ranking.
    """
    model.eval()

    all_probs = []

    for _ in range(length):
        # Calculate probabilities and track
        probs = model(inputs)
        all_probs.append(probs)

        # Add continuation to inputs to allow for further predictions
        continuation = torch.argmax(predictions, dim=1, keepdim=True)
        inputs = torch.cat([inputs, continuation], dim=1)

    return torch.tensor(all_probs)


def mean_reciprocal_ranking(model, input_seqs, goal_seqs, pred_length):
    # Shape of generated_probs is (num_seqs, pred_length, 240)
    generated_probs = generate_probs(model, input_seqs, pred_length)

    # Shape of goal_seqs is (num_seqs, pred_length)

    # Sort over dimension 2 so that they're sorted within the probability space (seq and length dims isolated)
    sorted_probs, sorted_indices = torch.sort(generated_probs, dim=2, descending=True)

    # Prep rankings tensor as all zeros
    rankings = torch.zeros((input_seqs.shape[0], pred_length))

    # Iterate over input sequences
    for i, (gen, goal) in enumerate(zip(sorted_indices, goal_seqs)):
        for j in range(pred_length):
            # Find the index of the target event in the sorted indices tensor, note position
            rankings[i, j] = torch.nonzero(torch.eq(gen[j], goal[j])).item()

    # Add one to rankings (to avoid division by zero) and reciprocate
    rankings = torch.reciprocal(rankings + 1)

    # Average rankings over the 0 (sequence) dimension to capture the mean for a given temporal position
    final_rankings = torch.mean(rankings, dim=0)
    return final_rankings


def test_mrr():
    goal_seqs = torch.tensor([[1, 2, 1, 0], [2, 1, 0, 2]])
    pred_length = 4
    generated_probs = torch.tensor([[[.1, .2, .3], [.5, .3, .9], [.4, .2, .8], [.7, .3, .2]],
                                    [[.3, .4, .5], [.6, .2, .9], [.7, .1, .2], [.4, .6, .5]]])

    sorted_probs, sorted_indices = torch.sort(generated_probs, dim=2, descending=True)

    rankings = torch.zeros((goal_seqs.shape[0], pred_length))
    for i, (gen, goal) in enumerate(zip(sorted_indices, goal_seqs)):
        for j in range(pred_length):
            rankings[i, j] = torch.nonzero(torch.eq(gen[j], goal[j])).item()

    # Add one to rankings (to avoid division by zero) and reciprocate
    rankings = torch.reciprocal(rankings + 1)

    final_rankings = torch.mean(rankings, dim=0)
    return final_rankings


if __name__ == '__main__':
    # model_to_test = "240_onehot_globalattn_infpred_k128_v512.model.model"
    # model_obj = persistence.unpickle_model(model_to_test).to(DEVICE)
    #
    # given_elems = 500
    # _, test_dataset = training_util.create_train_test(event_loader.load_dataset(event_loader.MAESTRO_EVENTS_2017_240), given=given_elems)
    #
    # predictions = make_predictions(model_obj, test_dataset)
    #
    # conf_mat = confusion_matrix(predictions, test_dataset.ys)
    # acc = simple_accuracy(predictions, test_dataset.ys)
    # f1 = f1_score(predictions, test_dataset.ys)
    #
    # print(f"Accuracy: {acc}")
    # print(f"F1 Score: {f1}")
    #
    # plt.matshow(conf_mat)
    # plt.show()
    print(test_mrr())
