from .utils import save_on_master
from nni.nas.pytorch.callbacks import Callback


class LRSchedulerCallback(Callback):
    """
    Calls scheduler on every epoch ends.

    Parameters
    ----------
    scheduler : LRScheduler
        Scheduler to be called.
    """
    def __init__(self, scheduler, mode="epoch"):
        super().__init__()
        assert mode == "epoch"
        self.scheduler = scheduler
        self.mode = mode

    def on_epoch_end(self, epoch):
        """
        Call ``self.scheduler.step()`` on epoch end.
        """
        self.scheduler.step(epoch)


class SaveCheckpointCallback(Callback):
    """
    Calls ``trainer.export()`` on every epoch ends.

    Parameters
    ----------
    checkpoint_dir : str
        Location to save checkpoints.
    """
    def __init__(self, output_dir, optimizer, lr_scheduler, loss_scaler, args):
        super().__init__()
        self.output_dir = output_dir
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_scaler = loss_scaler
        self.args = args

    def on_epoch_end(self, epoch):
        """
        Dump to ``/checkpoint_dir/epoch_{number}.pth.tar`` on every epoch end.
        ``DataParallel`` object will have their inside modules exported.
        """
        checkpoint_paths = [self.output_dir / 'checkpoint.pth']
        if hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        for checkpoint_path in checkpoint_paths:
            save_on_master({
                'model': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': self.loss_scaler.state_dict(),
                'args': self.args,
            }, checkpoint_path)