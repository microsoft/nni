import time
from collections import defaultdict, deque
import datetime
import random
import math
import sys
from timm.utils.model import unwrap_model
import torch
import json
from timm.utils import accuracy
import torch.distributed as dist
from ..interface import BaseOneShotTrainer

# _logger = logging.getLogger(__name__)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not self.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AFSupernetTrainer(BaseOneShotTrainer):
    """
    This trainer trains a supernet.

    Parameters
    ----------
    model : nn.Module
        The Supernet model.
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
    """
    def __init__(self, model, criterion, data_loader_train, data_loader_val,
                 optimizer, device, num_epochs, loss_scaler,
                 max_norm, model_ema, mixup_fn,
                 amp, teacher_model, teach_loss, 
                 choices, mode, retrain_config, max_accuracy, output_dir, lr_scheduler):
        self.model = model
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
        self.lr_scheduler = lr_scheduler


    def _is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True


    def _get_rank(self):
        if not self._is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()


    def _is_main_process(self):
        return self._get_rank() == 0


    def _sample_configs(self, choices):
        config = {}
        dimensions = ['mlp_ratio', 'num_heads']
        depth = random.choice(choices['depth'])
        for dimension in dimensions:
            config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

        config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

        config['layer_num'] = depth
        return config

    def _save_on_master(self, *args, **kwargs):
        if self._is_main_process():
            torch.save(*args, **kwargs)

    def _train_one_epoch(self, epoch):
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
                config = self._sample_configs(choices=self.choices)
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
        if self.output_dir and self._is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        self.lr_scheduler.step(epoch)
        checkpoint_paths = [self.output_dir / 'checkpoint.pth']
        if hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        for checkpoint_path in checkpoint_paths:
            self._save_on_master({
                'model': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': self.loss_scaler.state_dict(),
            }, checkpoint_path)

    def _validate_one_epoch(self, epoch):
        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'

        # switch to evaluation mode
        self.model.eval()
        if self.mode == 'super':
            config = self._sample_configs(choices=self.choices)
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
        if self.output_dir and self._is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    def fit(self):
        for epoch in range(self.num_epochs):
            self._train_one_epoch(epoch)
            self._validate_one_epoch(epoch)

    
    @torch.no_grad()
    def export(self):
        return None