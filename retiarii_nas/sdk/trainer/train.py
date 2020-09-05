import nni
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule


class TrainingModule(LightningModule):
    def __init__(self, model,
                 train_dataloader=None,
                 val_dataloader=None,
                 optimizer=None,
                 training_step=None,
                 validation_step=None,
                 test_step=None):
        super().__init__()
        self.model = model
        self.train_dataloader_fn = train_dataloader
        self.val_dataloader_fn = val_dataloader
        self.optimizer_fn = optimizer
        self.training_step_fn = training_step
        self.validation_step_fn = validation_step
        self.test_step_fn = test_step

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.training_step_fn(self, batch, batch_idx)

    def configure_optimizers(self):
        return self.optimizer_fn(self)

    def train_dataloader(self):
        return self.train_dataloader_fn(self)

    def validation_step(self, batch, batch_idx):
        return self.validation_step_fn(self, batch, batch_idx)

    def val_dataloader(self):
        return self.val_dataloader_fn(self)

    def validation_epoch_end(self, outputs):
        # FIXME: supports classification case only
        avg_acc = np.mean([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_acc': avg_acc}
        nni.report_intermediate_result(avg_acc)
        return {'val_acc': avg_acc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # FIXME: same as validation for now
        return self.validation_step_fn(self, batch, batch_idx)

    def test_dataloader(self):
        # FIXME: same as validation for now
        return self.val_dataloader_fn(self)

    def test_epoch_end(self, outputs):
        # FIXME: same as validation for now
        result_dict = self.validation_epoch_end(outputs)
        nni.report_final_result(result_dict['val_acc'])
        return result_dict


def train(model_cls, train_dataloader=None, val_dataloader=None,
          optimizer=None, training_step=None, validation_step=None, test_step=None):
    assert all(callable(t) for t in [train_dataloader, val_dataloader, optimizer,
                                     training_step, validation_step, test_step])
    model = model_cls()
    model = TrainingModule(model,
                           train_dataloader=train_dataloader,
                           val_dataloader=val_dataloader,
                           optimizer=optimizer,
                           training_step=training_step,
                           validation_step=validation_step,
                           test_step=test_step)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model)
    trainer.test()
