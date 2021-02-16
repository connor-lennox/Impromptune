from math import ceil

import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

from Data import event_loader
from Predictive.Training import training_util
from Predictive.Models.predictive_lstm import PredictiveLSTM
from Predictive.Models.predictive_relative_attention_model import PRAm
from Predictive.Models import model_persistence
from Predictive.Models.global_local_models import StackedModel, ParallelModel
from Predictive.Models.test_models import InformedTestModel


device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, samples, epochs=10, batch_size=32, given=16, workers=3, train_ratio=0.8):
    train_dataset, test_dataset = training_util.create_train_test(samples, train_ratio, given)
    y_weights = training_util.create_weights_for_dataset(train_dataset)
    sampler = WeightedRandomSampler(y_weights, len(y_weights))

    loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True)

    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    num_batches = ceil(len(train_dataset) // batch_size)
    for i in range(epochs):
        loss_sum = 0
        acc_sum = 0
        for batch_index, sample in enumerate(loader):
            print(f"\rEpoch {i+1} " + training_util.progress_string(batch_index+1, num_batches), end="")
            xs = sample[0].to(device)
            ys = sample[1].to(device)

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
    test_acc_sum = 0
    test_batches = ceil(len(test_dataset) // batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    for batch_index, sample in enumerate(test_loader):
        print("\rCalculating Generalization Error: " +
              training_util.progress_string(batch_index+1, test_batches), end="")
        xs = sample[0].to(device)
        ys = sample[1].to(device)
        result = model(xs)
        test_acc_sum += training_util.accuracy(result, ys)
    print(f"\n\tGeneralization Accuracy: {test_acc_sum/test_batches}")


if __name__ == '__main__':
    data = event_loader.load_dataset(event_loader.MAESTRO_EVENTS_MEDIUM)

    model_to_load = None

    # Either load model parameters from the loaded file, or use hard coded ones (new model)
    if model_to_load is not None:
        net = model_persistence.load_model(model_to_load)

    else:
        # k_d = 64            # Key dimension
        # v_d = 333           # Value dimension
        # e_d = 256           # Embedding dimension
        # r_d = 1024           # Relative cutoff
        # attn_layers = 1     # Number of intermediary relative attention layers
        # n_heads = 8         # Attention heads
        # use_onehot_embed = True    # Use one-hot embedding or learned embeddings
        # local_range = (128, 128)    # Local range for models using local attention
        #
        # # net = PRAm(key_dim=k_d, value_dim=v_d, embedding_dim=e_d, num_attn_layers=attn_layers, relative_cutoff=r_d)
        # net = StackedModel(embedding_dim=e_d, key_dim=k_d, value_dim=v_d, relative_cutoff=r_d, n_heads=n_heads,
        #                    use_onehot_embed=use_onehot_embed, local_range=local_range)
        net = InformedTestModel()

    save_model = False      # Whether or not to save the model after training
    pickle_model = True     # Whether or not to pickle model after training
    pickle_model_name = "onehot-localattn-relu-pred-relu-linear-k256-v512"

    net = net.to(device)
    train_model(net, data, batch_size=16, given=500, epochs=5)

    if save_model:
        model_persistence.save_model(net)

    if pickle_model:
        model_persistence.pickle_model(net, pickle_model_name)
