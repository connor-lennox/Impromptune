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
    # test_lstm = PredictiveLSTM()
    # test_lstm.load_state_dict(torch.load(r"C:\Users\Connor\Documents\Research\Impromptune\TrainedModels\test_lstm_1-2"))
    maestro_small = event_loader.load_dataset(event_loader.MAESTRO_EVENTS_SMALL_DENSE)
    # for i in range(3, 7):
    #     train_model(test_lstm, maestro_small, batch_size=1024, given=16, epochs=5)
    #     filename = f'test_lstm_1-{i}'
    #     with open(r"C:\Users\Connor\Documents\Research\Impromptune\TrainedModels" + '\\' + filename, 'wb+') as outfile:
    #         torch.save(test_lstm.state_dict(), outfile)
    test_pram = PRAm(key_dim=128, embedding_dim=256, use_onehot_embed=False, num_attn_layers=2)
    # test_pram.load_state_dict(torch.load(r"C:\Users\Connor\Documents\Research\Impromptune\TrainedModels\test_pram_k128_v256_0"))
    train_model(test_pram, maestro_small, batch_size=256, given=256, epochs=50)
    # filename = 'test_pram_k128_v256_1'
    # with open(r"C:\Users\Connor\Documents\Research\Impromptune\TrainedModels" + '\\' + filename, 'wb+') as outfile:
    #     torch.save(test_pram.state_dict(), outfile)
