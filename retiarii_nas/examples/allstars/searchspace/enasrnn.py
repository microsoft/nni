import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.nas.pytorch.mutables import LayerChoice, InputChoice

from blocks.rnn_utils import EmbeddingDropout, LockedDropout, create_mask2d


class Cell(nn.Module):
    def __init__(self, args):
        super(Cell, self).__init__()
        self.n_hidden = args.n_hidden
        self.dropout_h = args.dropout_rnn_h  # dropout for hidden nodes
        self.dropout_x = args.dropout_rnn_x  # dropout for input nodes
        self.steps = args.steps

        self.bns = None
        if args.state_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(args.n_hidden, affine=True) for _ in range(args.steps + 1)])

        self._W0 = nn.Parameter(torch.zeros((2 * args.n_hidden, 2 * args.n_hidden)))
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.zeros((args.n_hidden, 2 * args.n_hidden))) for _ in range(args.steps)
        ])
        self.activations = nn.ModuleList([LayerChoice([
            nn.Tanh(),
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Identity()
        ], key=f"act/{i}") for i in range(args.steps)])
        self.switch = nn.ModuleList([InputChoice(n_candidates=i + 1, n_chosen=1, key=f"switch/{i}") for i in range(args.steps)])
        self.reset_parameters(args.init_range)

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
        if self.bns is not None:
            s0 = self.bns[0](s0)
        states = [s0]
        for i in range(self.steps):
            s_prev = self.switch[i](states)  # (N, C)
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])
            else:
                ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.n_hidden, dim=-1)
            c = torch.sigmoid(c)
            h = self.activations[i](h)
            s = s_prev + c * (h - s_prev)
            if self.bns is not None:
                s = self.bns[i + 1](s)
            states.append(s)
        output = torch.mean(torch.stack(states[1:], -1), -1)
        return output


class RNNModel(nn.Module):
    """
    Encoder + Recurrent + Decoder
    """

    def __init__(self, args, n_tokens):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        # to remove words from embedding layers
        self.encoder = EmbeddingDropout(n_tokens, args.n_hidden, dropout=args.dropout_emb)
        self.rnn = Cell(args)
        self.decoder = nn.Linear(args.n_hidden, n_tokens)
        self.decoder.weight = self.encoder.weight  # bind decoder weight to encoder weight

        self.dropout_inp = args.dropout_inp  # to add dropout for input embedding layers
        self.dropout = args.dropout  # dropout added to normal layers
        self.n_hidden = args.n_hidden
        self.n_tokens = n_tokens

        self.reset_parameters(args.init_range)

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

        if self.training:
            return logits, hidden, rnn_h, dropped_rnn_h
        return logits, hidden

    def generate_hidden(self, batch_size):
        return torch.zeros((batch_size, self.n_hidden), device="cuda")
