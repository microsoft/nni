import os
import time

import nni
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch.backends.cudnn as cudnn
import random

class TrainingModule(LightningModule):
    def __init__(self, model,
                 train_dataloader=None,
                 val_dataloader=None,
                 optimizer=None,
                 training_step=None,
                 validation_step=None,
                 test_step=None,
                 use_fake_input=False,
                 use_mix_para=False,
                 mutator=None,
                 mix_para_idx=None,
                 n_model_parallel=None,
                 not_training=False,
                 has_mask=False,
                 enable_ff_profiling=False,  # ff_profiling= True to enable fast-forward profling, exit
                 ff_steps=100,  # profile for ff_steps
                 ff_skip_steps=100,  # skip ff_skip_steps for warm-up
                 ff_out='perf.out'):
        super().__init__()
        self.model = model
        self.train_dataloader_fn = train_dataloader
        self.val_dataloader_fn = val_dataloader
        self.optimizer_fn = optimizer
        self.training_step_fn = training_step
        self.validation_step_fn = validation_step
        self.test_step_fn = test_step
        self.use_fake_input = use_fake_input

        self.use_mix_para = use_mix_para
        self.mutator = mutator
        self.mix_para_idx = mix_para_idx
        self.n_model_parallel = n_model_parallel
        self.not_training = not_training
        self.has_mask = has_mask

        self.enable_ff_profiling = enable_ff_profiling
        self.ff_steps = ff_steps
        self.ff_skip_steps = ff_skip_steps
        assert(ff_skip_steps > 0)
        self.ff_out = ff_out
        self.ff_step_idx = 0

    def forward(self, *args, **kwargs):
        if self.use_fake_input:
            return self.model()
        else:
            return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        ret = self.training_step_fn(self, batch, batch_idx,
                                    use_mix_para=self.use_mix_para,
                                    mutator=self.mutator,
                                    n_model_parallel=self.n_model_parallel,
                                    not_training=self.not_training,
                                    has_mask=self.has_mask)

        if self.enable_ff_profiling:
            self.ff_step_idx += 1
            if self.ff_step_idx == self.ff_skip_steps:
                self.profile_start = time.time()
            if self.ff_step_idx - self.ff_skip_steps >= self.ff_steps:
                self.profile_end = time.time()
                with open(self.ff_out, 'w') as fp:
                    fp.write(
                        f"{self.profile_end - self.profile_start} {self.ff_steps}\n")
                    exit()
        return ret

    def configure_optimizers(self):
        if self.not_training:
            return None
        else:
            return self.optimizer_fn(self)

    def train_dataloader(self):
        return self.train_dataloader_fn(self)

    def validation_step(self, batch, batch_idx):
        return self.validation_step_fn(self, batch, batch_idx, use_mix_para=self.use_mix_para, mutator=self.mutator, not_training=self.not_training, has_mask=self.has_mask)

    def val_dataloader(self):
        return self.val_dataloader_fn(self)

    def validation_epoch_end(self, outputs):
        # FIXME: supports classification case only
        avg_acc = np.mean([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_acc': avg_acc}
        nni.report_intermediate_result(avg_acc)
        return {'val_acc': avg_acc}

    def test_step(self, batch, batch_idx):
        # FIXME: same as validation for now
        return self.validation_step_fn(self, batch, batch_idx, use_mix_para=self.use_mix_para, mutator=self.mutator)

    def test_dataloader(self):
        # FIXME: same as validation for now
        return self.val_dataloader_fn(self)

    def test_epoch_end(self, outputs):
        # FIXME: same as validation for now
        result_dict = self.validation_epoch_end(outputs)
        nni.report_final_result(result_dict['val_acc'])
        return result_dict


def train(model_cls, train_dataloader=None, val_dataloader=None,
          optimizer=None, training_step=None, validation_step=None,
          test_step=None, use_fake_input=False, use_mix_para=False,
          mutator_cls=None, not_training=False, has_mask=False,
          profile_out="", profile_steps=100, logger=False):
    assert all(callable(t) for t in [train_dataloader, val_dataloader, optimizer,
                                     training_step, validation_step, test_step])

    use_distributed = False
    mix_para_idx = 0
    n_model_parallel = 1
    self_rank = 0
    if 'is_distributed' in os.environ:
        use_distributed = True
        # torch.cuda.set_device(os.environ["rank"])
        world_size = int(os.environ["world_size"])
        backend = os.environ["distributed_backend"]
        self_rank = int(os.environ["rank"])
        torch.distributed.init_process_group(
            backend=backend, init_method=f'tcp://127.0.0.1:{12400}', rank=self_rank, world_size=world_size)
        torch.cuda.set_device(int(os.environ["DEVICE_ID"]))
        mix_para_idx = self_rank
        n_model_parallel = world_size
    # cudnn.benchmark = True
    model = model_cls()
    model.cuda()
    if use_distributed and use_mix_para:
        from apex.parallel import DistributedDataParallel

        if world_size > 1:
            for p in model.parameters():
                p.grad = torch.zeros_like(p)

        model = DistributedDataParallel(model, delay_allreduce=True)
        torch.distributed.barrier()

    mutator = None
    if use_mix_para:
        mutator = mutator_cls(model)
        for _ in range(self_rank):
            mutator.reset()

    enable_ff_profiling = False
    if profile_out != "":
        enable_ff_profiling = True
    model = TrainingModule(model,
                           train_dataloader=train_dataloader,
                           val_dataloader=val_dataloader,
                           optimizer=optimizer,
                           training_step=training_step,
                           validation_step=validation_step,
                           test_step=test_step,
                           use_fake_input=use_fake_input,
                           use_mix_para=use_mix_para,
                           mutator=mutator,
                           mix_para_idx=mix_para_idx,
                           n_model_parallel=n_model_parallel,
                           not_training=not_training,
                           has_mask=has_mask,
                           enable_ff_profiling=enable_ff_profiling,
                           ff_out=profile_out,
                           ff_steps=profile_steps
                           )
    # if resume or eval:
    #     assert(ckpt_dir!=None)
    # if ckpt_dir != None:
    #     if resume:
    #         assert('.ckpt' in ckpt_dir)
    #         for idx, m in enumerate(model.named_modules()):
    #             print(idx, '->', m)
    #         for p in model.parameters():
    #             print(p)
    #         model = model.load_from_checkpoint(ckpt_dir)
    #         print(model.learning_rate)
    #         trainer = Trainer(resume_from_checkpoint=ckpt_dir)
    #         if eval:
    #             trainer.test(model, test_dataloaders=model.val_dataloader())
    #     else:
    #         trainer = Trainer(max_epochs=60, logger=logger,
    #                         default_root_dir=ckpt_dir)
        
    # else:
    trainer = Trainer(max_epochs=1, logger=logger)
    # disable logger to avoid multiple jobs may use the same log dir
    trainer.fit(model)
    trainer.test()
