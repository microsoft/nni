import torch
import torch.nn as nn
import torch.nn.functional as F

from sdk.mutators import Mutator
from sdk.translate_code import gen_pytorch_graph
from .blocks import EmbeddingDropout, LockedDropout, create_mask2d


_ACTIVATIONS = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity
}


class Cell(nn.Module):
    def __init__(self, activations, inputs, n_hidden=128, dropout_rnn_h=0.5, dropout_rnn_x=0.5):
        super(Cell, self).__init__()
        self.activations = activations
        self.inputs = inputs
        self.n_hidden = n_hidden
        self.dropout_h = dropout_rnn_h  # dropout for hidden nodes
        self.dropout_x = dropout_rnn_x  # dropout for input nodes
        self.steps = len(activations)

        self._W0 = nn.Parameter(torch.zeros((2 * n_hidden, 2 * n_hidden)))
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.zeros((n_hidden, 2 * n_hidden))) for _ in range(self.steps)
        ])
        self.activations = nn.ModuleList([_ACTIVATIONS[a]() for a in self.activations])
        self.reset_parameters(0.5)

    def reset_parameters(self, init_range):
        self._W0.data.uniform_(-init_range, init_range)
        for p in self._Ws:
            p.data.uniform_(-init_range, init_range)

    def forward(self, inputs, hidden):
        L, N = inputs.size(0), inputs.size(1)

        if self.training:
            x_mask = create_mask2d(N, inputs.size(2), keep_prob=1 - self.dropout_x)
            h_mask = create_mask2d(N, hidden.size(1), keep_prob=1 - self.dropout_h)
        else:
            x_mask = h_mask = None

        hiddens = []
        for t in range(L):
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        return hiddens, hidden

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.n_hidden, dim=-1)
        c0 = torch.sigmoid(c0)
        h0 = torch.tanh(h0)
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def cell(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        states = [s0]
        for i in range(self.steps):
            s_prev = states[self.inputs[i]]  # (N, C)
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])
            else:
                ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.n_hidden, dim=-1)
            c = torch.sigmoid(c)
            h = self.activations[i](h)
            s = s_prev + c * (h - s_prev)
            states.append(s)
        output = torch.mean(torch.stack(states[1:], -1), -1)
        return output


class RNNModel(nn.Module):
    """
    Encoder + Recurrent + Decoder
    """

    def __init__(self, n_hidden=128, dropout_emb=0.5, dropout_inp=0.5, dropout=0.5, n_tokens=10000):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        # to remove words from embedding layers
        self.encoder = EmbeddingDropout(n_tokens, n_hidden, dropout=dropout_emb)
        self.rnn = Cell(['tanh'] * 8, [i for i in range(8)])
        self.decoder = nn.Linear(n_hidden, n_tokens)
        self.decoder.weight = self.encoder.weight  # bind decoder weight to encoder weight

        self.dropout_emb = dropout_emb
        self.dropout_inp = dropout_inp  # to add dropout for input embedding layers
        self.dropout = dropout  # dropout added to normal layers
        self.n_hidden = n_hidden
        self.n_tokens = n_tokens

        self.reset_parameters(0.5)

    def reset_parameters(self, init_range):
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inputs, hidden):
        batch_size = inputs.size(1)

        emb = self.encoder(inputs)  # (L, N, C)
        emb = self.lockdrop(emb, self.dropout_inp)  # (L, N, C)

        rnn_h, hidden = self.rnn(emb, hidden)  # (L, N, C), (N, C)
        dropped_rnn_h = self.lockdrop(rnn_h, self.dropout)  # (L, N, C)

        logits = self.decoder(dropped_rnn_h.view(-1, self.n_hidden))  # (L*N, Vocab)
        logits = logits.view(-1, batch_size, self.n_tokens)  # (L, N, Vocab)

        return logits, hidden

    def generate_hidden(self, batch_size):
        return torch.zeros((batch_size, self.n_hidden), device='cpu')


def nasrnn():
    model = RNNModel()
    cells = []
    for name, module in model.named_modules():
        if isinstance(module, Cell):
            cells.append(name)
    model(torch.randint(5, (30, 1)), torch.randn(1, 128))
    model_graph = gen_pytorch_graph(model,
                                    dummy_input=(torch.randint(5, (30, 1)),
                                                 torch.randn(1, 128)))
    mutators = []
    return model_graph, mutators
