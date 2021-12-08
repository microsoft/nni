import random
import math
import sys
from timm.utils.model import unwrap_model
import torch
import json
from timm.utils import accuracy
from .utils import sample_configs,MetricLogger,SmoothedValue,is_main_process
from nni.nas.pytorch.trainer import Trainer


class AFSupernetTrainer(Trainer):
    """
    This trainer trains a supernet.

    Parameters
    ----------
    model : nn.Module
        The Supernet model.
    mutator : Mutator
        A mutator object that has been initialized with the model.
    criterion : callable
        Called with logits and targets. Returns a loss tensor.
    data_loader_train : callable
        Data loader of training. Raise ``StopIteration`` when one epoch is exhausted.
    data_loader_val : iterablez
        Data loader of validation. Raise ``StopIteration`` when one epoch is exhausted.
    optimizer : Optimizer
        Optimizer that optimizes the model.
    device : torch.device
        The device type.
    num_epochs : int
        Number of epochs of training.
    loss_scaler : callable
        The loss scaler.
    max_norm : float
        The max gradients.
    model_ema : nn.Module
        The model with ema.
    mixup_fn : callable
        The mixup function.
    amp : Bool
        Indicator of using amp.
    teacher_model : nn.Module
        The teacher model.
    teacher_loss : callable
        Called with logits and targets. Returns a loss tensor.
    choices : dict
        The dict contains the config of the supernet.
    mode : str
        Training mode.
    retrain_config : dict
        The config of subnet.
    max_accuracy : float
        The maximum accuracy on validation set.
    output_dir : str
        The output path.
    callbacks : list of Callback
        Callbacks to plug into the trainer. See Callbacks.
    """
    def __init__(self, model, mutator, criterion, data_loader_train, data_loader_val,
                 optimizer, device, num_epochs, loss_scaler,
                 max_norm, model_ema, mixup_fn,
                 amp, teacher_model, teach_loss, 
                 choices, mode, retrain_config, max_accuracy, output_dir, callbacks):
        assert torch.cuda.is_available()
        super(AFSupernetTrainer, self).__init__(model, mutator, criterion, None,
                                                    optimizer, num_epochs, None, None,
                                                    None, None, device, None, callbacks)
        self.model = model
        self.mutator = mutator
        self.criterion = criterion
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.loss_scaler = loss_scaler
        self.max_norm = max_norm
        self.model_ema = model_ema
        self.mixup_fn = mixup_fn
        self.amp = amp
        self.teacher_model = teacher_model
        self.teach_loss = teach_loss
        self.choices = choices
        self.mode = mode
        self.retrain_config = retrain_config
        self.max_accuracy = max_accuracy
        self.output_dir = output_dir

    def train_one_epoch(self, epoch):
        self.data_loader_train.sampler.set_epoch(epoch)
        self.model.train()
        self.criterion.train()
        # set random seed
        random.seed(epoch)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10
        if self.mode == 'retrain':
            config = self.retrain_config
            model_module = unwrap_model(self.model)
            print(config)
            model_module.set_sample_config(config=config)
            print(model_module.get_sampled_params_numel(config))

        for samples, targets in metric_logger.log_every(self.data_loader_train, print_freq, header):
            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # sample random config
            if self.mode == 'super':
                config = sample_configs(choices=self.choices)
                model_module = unwrap_model(self.model)
                model_module.set_sample_config(config=config)
            elif self.mode == 'retrain':
                config = self.retrain_config
                model_module = unwrap_model(self.model)
                model_module.set_sample_config(config=config)
            if self.mixup_fn is not None:
                samples, targets = self.mixup_fn(samples, targets)
            if self.amp:
                with torch.cuda.amp.autocast():
                    if self.teacher_model:
                        with torch.no_grad():
                            teach_output = self.teacher_model(samples)
                        _, teacher_label = teach_output.topk(1, 1, True, True)
                        outputs = self.model(samples)
                        loss = 1/2 * self.criterion(outputs, targets) + 1/2 * self.teach_loss(outputs, teacher_label.squeeze())
                    else:
                        outputs = self.model(samples)
                        loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(samples)
                if self.teacher_model:
                    with torch.no_grad():
                        teach_output = self.teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    loss = 1 / 2 * self.criterion(outputs, targets) + 1 / 2 * self.teach_loss(outputs, teacher_label.squeeze())
                else:
                    loss = self.criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            self.optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            if self.amp:
                is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
                self.loss_scaler(loss, self.optimizer, clip_grad=self.max_norm,
                        parameters=self.model.parameters(), create_graph=is_second_order)
            else:
                loss.backward()
                self.optimizer.step()

            torch.cuda.synchronize()
            if self.model_ema is not None:
                self.model_ema.update(self.model)

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        train_status = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_status.items()},
                     'epoch': epoch,}
        if self.output_dir and is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    def validate_one_epoch(self, epoch):
        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'

        # switch to evaluation mode
        self.model.eval()
        if self.mode == 'super':
            config = sample_configs(choices=self.choices)
            model_module = unwrap_model(self.model)
            model_module.set_sample_config(config=config)
        else:
            config = self.retrain_config
            model_module = unwrap_model(self.model)
            model_module.set_sample_config(config=config)


        print("sampled model config: {}".format(config))
        parameters = model_module.get_sampled_params_numel(config)
        print("sampled model parameters: {}".format(parameters))

        for images, target in metric_logger.log_every(self.data_loader_val, 10, header):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            # compute output
            if self.amp:
                with torch.cuda.amp.autocast():
                    output = self.model(images)
                    loss = criterion(output, target)
            else:
                output = self.model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
        self.max_accuracy = max(self.max_accuracy, metric_logger.meters['acc1'].global_avg)
        print(f'Max accuracy: {self.max_accuracy:.2f}%')
        val_status = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'val_{k}': v for k, v in val_status.items()},
                     'epoch': epoch,}
        if self.output_dir and is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
