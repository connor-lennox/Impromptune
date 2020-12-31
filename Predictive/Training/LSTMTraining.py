import torch

from Data import event_loader
from Predictive.Training import training_util
from Predictive.Models.predictive_lstm import PredictiveLSTM


def train_lstm(lstm, samples, epochs=100, batch_size=32, given=32, train_ratio=0.8):
    x_train, y_train, x_test, y_test = training_util.create_train_test(samples, train_ratio, given)

    optim = torch.optim.Adam(lstm.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    num_batches = len(x_train) // batch_size
    for i in range(epochs):
        loss_sum = 0
        acc_sum = 0
        for batch_index, batch in enumerate(range(0, len(x_train)-1, batch_size)):
            print(f"\rEpoch {i+1} " + training_util.progress_string(batch_index+1, num_batches, ), end="")
            xs = x_train[batch:batch+batch_size]
            ys = y_train[batch:batch+batch_size]

            optim.zero_grad()
            result = lstm(xs)

            loss = criterion(result, ys)
            loss_sum += loss.item()
            acc_sum += training_util.accuracy(result, ys)

            loss.backward()
            optim.step()

        print(f"\n\tLoss={loss_sum/num_batches}, Acc={acc_sum/num_batches}")


if __name__ == '__main__':
    test_lstm = PredictiveLSTM()
    maestro_small = event_loader.load_dataset(event_loader.MAESTRO_EVENTS_SMALL_DENSE)
    train_lstm(test_lstm, maestro_small)
