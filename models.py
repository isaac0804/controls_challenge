import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, d_input=4, d_model=64, num_layers=4, nhead=4) -> None:
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.projection_up = nn.Linear(d_input, d_model)
        self.projection_down = nn.Linear(d_model, 1)

        self.conv1 = nn.Conv1d(d_model, d_model*4, kernel_size=15, padding=7, padding_mode="replicate")
        self.conv2 = nn.Conv1d(d_model*4, d_model, kernel_size=15, padding=7, padding_mode="replicate")
        self.conv3 = nn.Conv1d(d_model, d_model*4, kernel_size=15, padding=7, padding_mode="replicate")
        self.conv4 = nn.Conv1d(d_model*4, d_model, kernel_size=15, padding=7, padding_mode="replicate")

        self.positonal_encoding = PositionalEncoding(d_model, dropout=0.1, max_len=100)
        self.dropout = nn.Dropout(p=0.1)

        norm_layer = nn.LayerNorm(d_model, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, bias=False, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm_layer)

        for name, param in self.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

    def forward(self, x):
        x = self.projection_up(x) * math.sqrt(self.d_model)

        x = x.permute([1, 2, 0]) # [N, B, D] -> [B, D, N]
        x = self.conv2(F.tanh(self.conv1(x))) + x
        x = x.permute([2, 0, 1]) # [B, D, N] -> [N, B, D] 

        x = self.positonal_encoding(x)
        x = self.transformer_encoder(x)

        x = x.permute([1, 2, 0]) # [N, B, D] -> [B, D, N]
        x = self.conv4(F.tanh(self.conv3(x))) + x
        x = x.permute([2, 0, 1]) # [B, D, N] -> [N, B, D] 

        x = self.projection_down(self.dropout(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # x = self.norm1(x)
        # x = self.self_attn(x, x, x, attn_mask=mask)[0] + x
        x = self.norm1(self.self_attn(x, x, x, attn_mask=mask)[0]) + x
        x = self.dropout(x)
        # x = self.norm2(x)
        # x = self.ff(x) + x
        x = self.norm2(self.ff(x)) + x
        return x

class Decoder(nn.Module):
    def __init__(self, d_input=5, d_model=64, num_layers=4, nhead=4, seq_len=100, dropout=0.05) -> None:
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.projection_up = nn.Linear(d_input, d_model)
        self.projection_down = nn.Linear(d_model, 1)
        self.projection_down = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.positonal_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)
        self.transformer_decoder = nn.ModuleList([DecoderBlock(d_model,nhead,dropout=dropout)]*num_layers)
        self.norm1 = nn.BatchNorm1d(d_input)

        for name, param in self.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size, device="cuda") == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask

    def forward(self, x):
        x = self.norm1(x.permute([1,2,0])).permute([2,0,1])
        x = self.projection_up(x)
        x = self.positonal_encoding(x)
        mask = self.get_tgt_mask(x.shape[0])
        for layer in self.transformer_decoder:
            x = layer(x, mask=mask)
        x = self.projection_down(x)
        return x