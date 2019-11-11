import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutator import PyTorchMutator


class StackedLSTMCell(nn.Module):
    def __init__(self, layers, size, bias):
        super().__init__()
        self.lstm_num_layers = layers
        self.lstm_modules = nn.ModuleList([nn.LSTMCell(size, size, bias=bias)
                                           for _ in range(self.lstm_num_layers)])

    def forward(self, inputs, hidden):
        prev_c, prev_h = hidden
        next_c, next_h = [], []
        for i, m in enumerate(self.lstm_modules):
            curr_c, curr_h = m(inputs, (prev_c[i], prev_h[i]))
            next_c.append(curr_c)
            next_h.append(curr_h)
            inputs = curr_h[-1]
        return next_c, next_h


class EnasMutator(PyTorchMutator):
    def __init__(self, model, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5, anchor_extra_step=False,
                 skip_target=0.4):
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.max_layer_choice = 0
        self.anchor_extra_step = anchor_extra_step
        self.skip_target = skip_target
        super().__init__(model)

    def before_build(self, model):
        self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
        self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]), requires_grad=False)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def after_build(self, model):
        self.embedding = nn.Embedding(self.max_layer_choice + 1, self.lstm_size)
        self.soft = nn.Linear(self.lstm_size, self.max_layer_choice)

    def before_pass(self):
        super().before_pass()
        self._anchors_hid = dict()
        self._selected_layers = []
        self._selected_inputs = []
        self._inputs = self.g_emb.data
        self._c = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self._h = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self.sample_log_prob = 0
        self.sample_entropy = 0
        self.sample_skip_penalty = 0

    def _lstm_next_step(self):
        self._c, self._h = self.lstm(self._inputs, (self._c, self._h))

    def _mark_anchor(self, key):
        self._anchors_hid[key] = self._h[-1]

    def on_init_layer_choice(self, mutable):
        if self.max_layer_choice == 0:
            self.max_layer_choice = mutable.length
        assert self.max_layer_choice == mutable.length, \
            "ENAS mutator requires all layer choice have the same number of candidates."

    def on_calc_layer_choice_mask(self, mutable):
        self._lstm_next_step()
        logit = self.soft(self._h[-1])
        if self.tanh_constant is not None:
           logit = self.tanh_constant * torch.tanh(logit)
        branch_id = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
        log_prob = self.cross_entropy_loss(logit, branch_id)
        self.sample_log_prob += log_prob
        entropy = (log_prob * torch.exp(-log_prob)).detach()
        self.sample_entropy += entropy
        self._inputs = self.embedding(branch_id)
        self._selected_layers.append(branch_id.item())
        return F.one_hot(branch_id).bool().view(-1)

    def on_calc_input_choice_mask(self, mutable, semantic_labels):
        if mutable.n_selected is None:
            query, anchors = [], []
            for label in semantic_labels:
                if label not in self._anchors_hid:
                    self._lstm_next_step()
                    self._mark_anchor(label)  # empty loop, fill not found
                query.append(self.attn_anchor(self._anchors_hid[label]))
                anchors.append(self._anchors_hid[label])
            query = torch.cat(query, 0)
            query = torch.tanh(query + self.attn_query(self._h[-1]))
            query = self.v_attn(query)
            logit = torch.cat([-query, query], 1)
            if self.tanh_constant is not None:
                logit = self.tanh_constant * torch.tanh(logit)

            skip = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip_prob = torch.sigmoid(logit)
            kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl

            log_prob = self.cross_entropy_loss(logit, skip)
            self.sample_log_prob += torch.sum(log_prob)
            entropy = (log_prob * torch.exp(-log_prob)).detach()
            self.sample_entropy += torch.sum(entropy)

            self.inputs = torch.matmul(skip.float(), torch.cat(anchors, 0)) / (1. + torch.sum(skip))
            self._selected_inputs.append(skip)
            return skip.bool()
        else:
            assert mutable.n_selected == 1, "Input choice must select exactly one or any in ENAS."
            raise NotImplementedError

    def exit_mutable_scope(self, mutable_scope):
        self._mark_anchor(mutable_scope.key)
