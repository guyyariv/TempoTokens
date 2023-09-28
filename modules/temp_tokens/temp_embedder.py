import torch.nn as nn
from modules.fga.atten import Atten
import torch
import math


class TempEmbedder(nn.Module):
    def __init__(self, input_size=128, hidden_size=768, output_size=1024):
        super(TempEmbedder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()
        self.fga = Atten(util_e=[output_size], pairwise_flag=False)
        # self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, audio_embs):
        audio_embs = self.fc1(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc2(audio_embs)
        attend = self.fga([audio_embs])[0]
        # attend = self.gelu(attend)
        # output = self.fc3(attend)
        # return output
        return attend


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_seq_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
