# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class PosEncoding(nn.Module):
    def __init__(self, hidden_dim, max_seq_len=80):
        super().__init__()
        self.hidden_dim = hidden_dim

        pe = torch.zeros(max_seq_len, hidden_dim)
        for pos in range(max_seq_len):
            for i in range(0, hidden_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / hidden_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / hidden_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.hidden_dim)
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -1e9)
    attention_map = F.softmax(logits, dim=-1)
    if dropout is not None:
        attention_map = dropout(attention_map)
    return torch.matmul(attention_map, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.n_heads = n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # project and reshaping
        k_project = self.k_proj(key)
        q_project = self.q_proj(query)
        v_project = self.v_proj(value)
        k_reshape = k_project.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        q_reshape = q_project.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v_reshape = v_project.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # merge heads and output
        scores = attention(q_reshape, k_reshape, v_reshape, mask, self.dropout)
        scores = scores.transpose(1, 2).contiguous()
        scores = scores.view(batch_size, -1, self.hidden_dim)

        return self.output_proj(scores)


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim=2048, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, intermediate_dim)
        self.dense2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dense2(self.dropout(F.relu(self.dense1(x))))


class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.alpha = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(hidden_dim, n_heads)
        self.ff_layer = FeedForwardLayer(hidden_dim)

        self.norm1 = LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inp, mask):
        x = self.norm1(inp)
        x = inp + self.dropout1(self.self_attn(x, x, x, mask))
        x = x + self.dropout2(self.ff_layer(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(hidden_dim, n_heads)
        self.cross_attn = MultiHeadAttention(hidden_dim, n_heads)
        self.ff = FeedForwardLayer(hidden_dim)

        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inp, mask, encoder_output, encoder_output_mask):
        x = self.norm1(inp)
        x = inp + self.dropout1(self.self_attn(x, x, x, mask))
        x = x + self.dropout2(self.cross_attn(self.norm2(x), encoder_output, encoder_output, encoder_output_mask))
        x = x + self.dropout3(self.ff(self.norm3(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_layers, hidden_dim, n_heads):
        super().__init__()

        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.posencoding = PosEncoding(hidden_dim)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderLayer(n_heads, hidden_dim)) for _ in range(n_layers)])
        self.layernorm = LayerNorm(hidden_dim)

    def forward(self, src, mask):
        x = self.embedding(src)
        x = self.posencoding(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.layernorm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_layers, hidden_dim, n_heads):
        super().__init__()

        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.posencoding = PosEncoding(hidden_dim)
        self.layers = nn.ModuleList([copy.deepcopy(TransformerDecoderLayer(n_heads, hidden_dim)) for _ in range(n_layers)])
        self.layernorm = LayerNorm(hidden_dim)

    def forward(self, inp, mask, encoder_output, encoder_output_mask):
        x = self.embedding(inp)
        x = self.posencoding(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask, encoder_output, encoder_output_mask)
        return self.layernorm(x)


class TransformerForSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, n_layers, hidden_dim, n_heads):
        super().__init__()

        self.encoder = TransformerEncoder(src_vocab_size, n_layers, hidden_dim, n_heads)
        self.decoder = TransformerDecoder(tgt_vocab_size, n_layers, hidden_dim, n_heads)
        self.output_dense = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_outputs = self.encoder(src, src_mask)
        decoder_outputs = self.decoder(tgt, tgt_mask, encoder_outputs, src_mask)

        return self.output_dense(decoder_outputs)
