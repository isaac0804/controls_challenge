import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

class MyMLP(nn.Module):
    def __init__(self, d_input, d_hidden, d_out):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_hidden)
        self.fc4 = nn.Linear(d_hidden, d_hidden)
        self.fc5 = nn.Linear(d_hidden, d_hidden)
        self.fc6 = nn.Linear(d_hidden, d_out)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc3(torch.tanh(self.fc2(x))) + x
        x = self.dp1(x)
        x = self.fc5(torch.tanh(self.fc4(x))) + x
        x = self.dp2(x)
        x = self.fc6(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # shape : [seq_len, batch_size, embedding_dim]
        # if x.shape[0] == 100:
        #     import time
        #     plt.imshow(self.pe[:x.size(0)].squeeze(),)
        #     plt.imshow(x[:,0,:])
        #     plt.show()
        #     time.sleep(100)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, d_input=4, d_model=64, num_layers=4, nhead=4) -> None:
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.projection_up = nn.Linear(d_input, d_model)
        norm_layer = nn.LayerNorm(d_model, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, bias=False, norm_first=True)
        self.positonal_encoding = PositionalEncoding(d_model, dropout=0.1, max_len=100)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm_layer)
        self.projection_down = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(p=0.1)

        for name, param in self.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

    def forward(self, x):
        x = self.projection_up(x) * math.sqrt(self.d_model)
        x = self.positonal_encoding(x)
        x = self.transformer_encoder(x)
        x = self.projection_down(self.dropout(x))
        return x

