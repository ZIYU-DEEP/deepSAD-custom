import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class LSTM_Net(BaseNet):

    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.lstm1 = nn.GRU(128, 128, batch_first=True)
#         self.lstm2 = nn.GRU(64, 32, batch_first=True)
#         self.lstm3 = nn.GRU(32, 16, batch_first=True)
        self.fc1 = nn.Linear(128 * 1000, self.rep_dim, bias=False)

    def forward(self, x):
        x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
#         x, _ = self.lstm3(x)
        x = x.reshape(x.size(0), 128 * 1000).contiguous()
        x = self.fc1(x)
        return x


class LSTM_Decoder(BaseNet):

    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim

        # Decoder network
        self.lstm1 = nn.GRU(128, 128, batch_first=True)
#         self.lstm2 = nn.GRU(32, 64, batch_first=True)
#         self.lstm3 = nn.GRU(64, 128, batch_first=True)
        self.fc1 = nn.Linear(self.rep_dim, 128 * 1000)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), 1000, 128).contiguous()
        x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
#         x, _ = self.lstm3(x)
        x = torch.sigmoid(x)
        return x


class LSTM_Autoencoder(BaseNet):
    def __init__(self, rep_dim=2):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = LSTM_Net(rep_dim=rep_dim)
        self.decoder = LSTM_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
