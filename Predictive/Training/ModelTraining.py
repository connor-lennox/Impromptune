import datetime
import re

import torch

from Data import event_loader
from Predictive.Training import training_util
from Predictive.Models.predictive_lstm import PredictiveLSTM
from Predictive.Models.predictive_relative_attention_model import PRAm
from Predictive.Models import model_persistence
from Predictive.Models.global_local_models import StackedModel, ParallelModel


def train_model(model, samples, epochs=10, batch_size=32, given=16, train_ratio=0.8):
    x_train, y_train, x_test, y_test = training_util.create_train_test(samples, train_ratio, given)
    # x_train, y_train = x_train[:512], y_train[:512]

    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    num_batches = len(range(0, len(x_train)-1, batch_size))
    for i in range(epochs):
        loss_sum = 0
        acc_sum = 0
        for batch_index, batch in enumerate(range(0, len(x_train)-1, batch_size)):
            print(f"\rEpoch {i+1} " + training_util.progress_string(batch_index+1, num_batches), end="")
            xs = x_train[batch:batch+batch_size]
            ys = y_train[batch:batch+batch_size]

            optim.zero_grad()
            result = model(xs)

            loss = criterion(result, ys)
            loss_sum += loss.item()
            acc_sum += training_util.accuracy(result, ys)

            loss.backward()
            optim.step()

        print(f"\n\tLoss={loss_sum/num_batches}, Acc={acc_sum/num_batches}")

    print()
    # Calculate accuracy over test data
    test_batches = len(range(0, len(x_test)-1, batch_size))
    test_acc_sum = 0
    for batch_index, batch in enumerate(range(0, len(x_test) - 1, batch_size)):
        print("\rCalculating Generalization Error: " +
              training_util.progress_string(batch_index+1, test_batches), end="")
        xs = x_test[batch:batch+batch_size]
        ys = y_test[batch:batch+batch_size]
        result = model(xs)
        test_acc_sum += training_util.accuracy(result, ys)
    print(f"\n\tGeneralization Accuracy: {test_acc_sum/test_batches}")


if __name__ == '__main__':
    data = event_loader.load_dataset(event_loader.MAESTRO_EVENTS_SMALL_DENSE)

    model_to_load = None

    # Either load model parameters from the loaded file, or use hard coded ones (new model)
    if model_to_load is not None:
        net = model_persistence.load_model(model_to_load)

    else:
        k_d = 64            # Key dimension
        v_d = 333           # Value dimension
        e_d = 256           # Embedding dimension
        r_d = 1024           # Relative cutoff
        attn_layers = 1     # Number of intermediary relative attention layers
        n_heads = 8         # Attention heads
        use_onehot_embed = True    # Use one-hot embedding or learned embeddings
        local_range = (128, 128)    # Local range for models using local attention

        # net = PRAm(key_dim=k_d, value_dim=v_d, embedding_dim=e_d, num_attn_layers=attn_layers, relative_cutoff=r_d)
        net = ParallelModel(embedding_dim=e_d, key_dim=k_d, value_dim=v_d, relative_cutoff=r_d, n_heads=n_heads,
                            use_onehot_embed=use_onehot_embed, local_range=local_range)

    save_model = True  # Whether or not to save the model after training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    train_model(net, data, batch_size=4, given=500, epochs=5)

    if save_model:
        model_persistence.save_model(net)
