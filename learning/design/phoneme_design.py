import torch.nn as nn

# model parameters
lay = 2
dim = 256
drop = 0.2
num_char = 49

# define the model
class PhonemeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=dim, num_layers=lay, batch_first=True, dropout=drop)
        self.dropout = nn.Dropout(drop)
        self.linear = nn.Linear(dim, num_char) # hidden dimensions as input, 49 possible outputs

    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
