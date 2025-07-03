import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class IQEmbedding(nn.Module):
    def __init__(self, word_size, embed_size):
        super(IQEmbedding, self).__init__()
        self.fc = nn.Linear(word_size, embed_size)

    def forward(self, x):
        x = x.to(self.fc.weight.dtype)
        x = self.fc(x)
        return F.relu(x)


class BertEmbedding(nn.Module):
    def __init__(self, word_size, embed_size, dropout=0):
        super(BertEmbedding, self).__init__()
        self.position = PositionalEmbedding(d_model=512)
        self.iq = IQEmbedding(word_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        # x = self.position(sequence) + self.iq(sequence)
        x = self.iq(sequence)
        return self.dropout(x)



