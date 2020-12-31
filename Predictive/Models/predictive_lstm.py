import torch.nn as nn
import torch.nn.functional as func


class PredictiveLSTM(nn.Module):
    def __init__(self, events_length=333, embed_dim=64, hidden_size=512):
        super(PredictiveLSTM, self).__init__()

        self.embedding = nn.Embedding(events_length, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=3)
        self.linear = nn.Linear(hidden_size, events_length)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        # Extract the last output of the LSTM (most informed)
        x = x[:, -1, :]
        # x = func.softmax(self.linear(x), dim=1)
        x = self.linear(x)
        return x
