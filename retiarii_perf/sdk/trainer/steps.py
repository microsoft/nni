import torch
import torch.nn.functional as F
from torch.autograd import Variable


def classifier_train_step(trainer, batch, batch_idx, use_mix_para=False,
                          mutator=None, n_model_parallel=None, not_training=False,
                          has_mask=False):
    if use_mix_para:
        # skip networks of this batch
        for _ in range(n_model_parallel):
            mutator.reset()
    if has_mask:
        x, mask, y = batch
    else:
        x, y = batch
    # zhenhua: trainer may change x, y due to cross-graph optimization
    if not_training:
        if has_mask:
            trainer(x, mask, y)
        else:
            trainer(x, y)
        # feed fake loss to non-training process
        y_hat = torch.randn((y.size()[0], 5))
        if has_mask:
            y_hat = y_hat.cuda()
        loss = F.cross_entropy(y_hat, y)
        loss = Variable(loss, requires_grad=True)
        #tensorboard_logs = {'train_loss': loss}
        return {"loss": loss}  # {'loss': loss, 'log': tensorboard_logs}
    else:
        if has_mask:
            x, y, y_hat = trainer(x, mask, y)
        else:
            x, y, y_hat = trainer(x, y)
        _, predicted = torch.max(y_hat.data, 1)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss}


def classifier_validation_test_step(trainer, batch, batch_idx, use_mix_para=False,
                                    mutator=None, not_training=False, has_mask=False):
    if use_mix_para:
        mutator.reset()
    if has_mask:
        x, mask, y = batch
    else:
        x, y = batch
    if not_training:
        if has_mask:
            trainer(x, mask, y)
        else:
            trainer(x, y)
        # feed fake loss to non-training process
        #_, predicted = torch.max(y.data, 1)
        accuracy = (y == y).sum().item() / y.size(0)
        return {'val_acc': accuracy}
    else:
        # zhenhua: trainer may change x, y due to cross-graph optimization
        if has_mask:
            x, y, y_hat = trainer(x, mask, y)
        else:
            x, y, y_hat = trainer(x, y)
        _, predicted = torch.max(y_hat.data, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
        return {'val_loss': F.cross_entropy(y_hat, y), 'val_acc': accuracy}
