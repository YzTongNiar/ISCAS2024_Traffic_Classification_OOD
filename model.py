import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=100, num_layers=2, output_dim=5,
                 dropout=0):
        """
        input_dim = number of features at each time step
                    (number of features given to each LSTM cell)
        hidden_dim = number of features produced by each LSTM cell (in each layer)
        num_layers = number of LSTM layers
        output_dim = number of classes of the floor texture
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False,)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, X_seq_len):

        hidden_features, (_, _) = self.lstm(X)  # (h_0, c_0) default to zeros
        out_pad, out_len = pad_packed_sequence(hidden_features, batch_first=True)
        lstm_out_forward = out_pad[range(len(out_pad)), X_seq_len-1, :self.hidden_dim]

        # lstm_out_backward = out_pad[range(len(out_pad)), X_seq_len-1, self.hidden_dim:]
        # lstm_out = torch.cat((lstm_out_forward, lstm_out_backward), 1)
        # out = self.fc(F.relu(self.bn(lstm_out_forward), inplace=True))

        out = self.fc(lstm_out_forward)
        return out, lstm_out_forward


