import datetime
import re

import torch

from Data import event_loader
from Predictive.Training import training_util
from Predictive.Models.predictive_lstm import PredictiveLSTM
from Predictive.Models.predictive_relative_attention_model import PRAm


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

    model_to_load = "1609810493_pram_k64_v256_e256_r128_attn1.pram"
    model_regex = r'\d+_pram_k(\d+)_v(\d+)_e(\d+)_r(\d+)_attn(\d+)\.pram'

    # Either load model parameters from the loaded file, or use hard coded ones (new model)
    if model_to_load is not None:
        match = re.match(model_regex, model_to_load)

        k_d = int(match[1])
        v_d = int(match[2])
        e_d = int(match[3])
        r_d = int(match[4])
        attn_layers = int(match[5])

    else:
        k_d = 64            # Key dimension
        v_d = 256           # Value dimension
        e_d = 256           # Embedding dimension
        r_d = 128           # Relative cutoff
        attn_layers = 1     # Number of intermediary relative attention layers

    save_model = True  # Whether or not to save the model after training

    net = PRAm(key_dim=k_d, value_dim=v_d, embedding_dim=e_d, num_attn_layers=attn_layers, relative_cutoff=r_d)

    if model_to_load is not None:
        net.load_state_dict(torch.load(r"TrainedModels\\" + model_to_load))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    train_model(net, data, batch_size=32, given=256, epochs=5)
    filename = f'{int(datetime.datetime.now().timestamp())}_pram_k{k_d}_v{v_d}_e{e_d}_r{r_d}_attn{attn_layers}.pram'

    if save_model:
        with open(r"TrainedModels" + '\\' + filename, 'wb+') as outfile:
            torch.save(net.state_dict(), outfile)
