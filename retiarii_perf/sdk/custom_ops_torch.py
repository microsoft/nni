import torch.nn as nn
import torch
class View(nn.Module):
    def __init__(self, shape=(-1, )):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Broadcast(nn.Module):
    def __init__(self, is_src, src_rank, size = None, dtype = None, device = None):
        super().__init__()
        self.is_src = is_src
        self.src_rank = src_rank
        self.size = size
        self.dtype = dtype
        self.device = device

    def forward(self, x = None):
        if self.is_src:
            torch.distributed.broadcast(x, self.src_rank)
            return x
        else:
            x = torch.zeros(self.size, dtype=self.dtype, device=self.device)
            torch.distributed.broadcast(x, self.src_rank)
            return x

class Bert(nn.Module):
    def __init__(self, pretrain_model, max_length, pad_to_max_length, truncation):
        super().__init__()
        pretrain_model = pretrain_model
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.truncation = truncation
        
        from transformers import BertModel, BertTokenizer
        self.model = BertModel.from_pretrained(pretrain_model).cuda()
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_model)

    def forward(self, x):
        text = x # 
        mask = text > 0
        with torch.no_grad():
            output, _ = self.model(text)
        mask = mask.float()
        mask = mask.bool()
        return output, mask

class Transpose(nn.Module):
    def __init__(self, dim0=0, dim1=1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)

class Expand(nn.Module):
    def __init__(self, num_copies):
        super().__init__()
        self.num_copies = num_copies
        
    def forward(self, x):
        return torch.cat([x] * self.num_copies)


class BatchSizeView(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        
    def forward(self, x):
        bs, _, s1, s2 = x.size()
        return x.view(self.batch_size, -1, s1, s2)

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.squeeze(x)

class FixedInputChoice(nn.Module):
    def __init__(self, choice_vector, gen=False):
        super(FixedInputChoice, self).__init__()
        self.choice_vector = choice_vector
        self.gen = gen

    def forward(self, inputs):
        #print("FixInputChoice, inputs" , [_.size() for _ in inputs])
        if self.gen:
            if len(inputs) == 1:
                return inputs[0]
            out = sum(inputs)
            # out = inputs[0]
            # for choice in inputs[1:]:
            #     out += choice
        else:
            #print("FixInputChoice", [(a,x) for a,x in zip(self.choice_vector, inputs)])
            out = sum([x for a, x in zip(self.choice_vector, inputs) if a])
            #print(out)
        #print("FixInputChoice, out" , out.size())
        return out