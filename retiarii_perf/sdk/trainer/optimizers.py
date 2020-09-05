import torch.optim as optim


def pytorch_builtin_optimizers(classname, **kwargs):
    def optimizer(trainer):
        return getattr(optim, classname)(trainer.parameters(), **kwargs)
    return optimizer
