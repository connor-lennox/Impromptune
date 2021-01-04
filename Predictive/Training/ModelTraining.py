import datetime

import torch

from Data import event_loader
from Predictive.Training import training_util
from Predictive.Models.predictive_lstm import PredictiveLSTM
from Predictive.Models.predictive_relative_attention_model import PRAm


def train_model(model, samples, epochs=10, batch_size=32, given=16, train_ratio=0.8):
    x_train, y_train, x_test, y_test = training_util.create_train_test(samples, train_ratio, given)
    x_train, y_train = x_train[:512], y_train[:512]

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


if __name__ == '__main__':
    data = event_loader.load_dataset(event_loader.MAESTRO_EVENTS_SMALL_DENSE)

    # Define model parameters
    k_d = 64            # Key dimension
    v_d = 256           # Value dimension
    e_d = 256           # Embedding dimension
    r_d = 128           # Relative cutoff
    attn_layers = 2     # Number of intermediary relative attention layers

    net = PRAm(key_dim=k_d, value_dim=v_d, embedding_dim=e_d, num_attn_layers=attn_layers, relative_cutoff=r_d)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    train_model(net, data, batch_size=64, given=256, epochs=50)
    filename = f'{int(datetime.datetime.now().timestamp())}_pram_k{k_d}_v{v_d}_e{e_d}_r{r_d}_attn{attn_layers}.pram'
    with open(r"TrainedModels" + '\\' + filename, 'wb+') as outfile:
        torch.save(net.state_dict(), outfile)
