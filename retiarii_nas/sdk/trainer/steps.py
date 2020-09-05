import torch
import torch.nn.functional as F


def classifier_train_step(trainer, batch, batch_idx):
    x, y = batch
    y_hat = trainer(x)
    loss = F.cross_entropy(y_hat, y)
    tensorboard_logs = {'train_loss': loss}
    return {'loss': loss, 'log': tensorboard_logs}


def classifier_validation_test_step(trainer, batch, batch_idx):
    x, y = batch
    y_hat = trainer(x)
    _, predicted = torch.max(y_hat.data, 1)
    accuracy = (predicted == y).sum().item() / y.size(0)
    return {'val_loss': F.cross_entropy(y_hat, y), 'val_acc': accuracy}
