from logging import error, warning
import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
from collections import OrderedDict
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from nni.retiarii.oneshot.pytorch.utils import replace_input_choice, replace_layer_choice
from pytorch_lightning.utilities.enums import DistributedType
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
import nni.nas.pytorch.mutables as mutables

class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, recurse):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]
    
    def r_forward(self, x):
        return self.op_choices.values()[0](x)

class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, recurse):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]
    
    def r_forward(self, x):
        return super().r_forward(x)

class DartsRepeat(nn.Module):
    pass

class DartsDataset(Dataset):
    def __init__(self, train_data = [Dataset, DataLoader], val_data = [Dataset, DataLoader]):
        super().__init__()
        # 是不是要两个取短的？但要是他把 val_dataset 取的很短不就很可惜？
        # concat dataset
        train_dataset = train_data if isinstance(train_data, Dataset) else train_data.dataset
        val_dataset = val_data if isinstance(val_data, Dataset) else val_data.dataset
        self.tds = train_dataset
        self.vds = val_dataset
        if len(train_dataset) / len(val_dataset) > 1.5 or len(val_dataset) / len(train_dataset) > 1.5 :
            warning('The length difference betweeen the training dataset({}) and the validation dataset({}) ' \
                'is too large, which will result in a waste of the longer one because the DARTS will truncate it.'.format(
                    len(train_dataset),len(val_dataset)
                ))
    
    def __getitem__(self, index):
        return self.tds[index], self.vds[index]
    
    def __len__(self):
        return min(len(self.tds), len(self.vds))

# val 改成 function 
CHOICE_REPLACE_DICT = {
    LayerChoice : DartsLayerChoice,
    InputChoice : DartsInputChoice
}

class DartsModel(pl.LightningModule):
    # 传入一个 hook 列表，每个 hook if-else 替换？
    # 传入 dict？ 可以参考一下标准库的实现，json？
    '''
        choice_replace_dict 是一个 xxxChoice 类到用户自己实现的 ChoiceReplace 的类的字典。可以改个名字。
    '''
    def __init__(self, model, arc_lr = 3.e-4, unrolled = False, custom_choice_replace_dict = None):
        super().__init__()
        assert isinstance(model, pl.LightningModule)
        self.model = model
        self.nas_modules = []
        choice_replace_dict = dict(CHOICE_REPLACE_DICT)
        breakpoint()
        if custom_choice_replace_dict is not None:
            for k,v in custom_choice_replace_dict.items():
                assert isinstance(v, nn.Module)
                choice_replace_dict[k] = v
#        for k,v in choice_replace_dict.items():
#            replace_module_with_type(self.model, choice_replace_dict[k], k, self.nas_modules)
        replace_layer_choice(self.model, choice_replace_dict[LayerChoice], self.nas_modules)
        replace_input_choice(self.model, choice_replace_dict[InputChoice], self.nas_modules)
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), arc_lr, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled
        self.model_optim = self.model.configure_optimizers()
    
    def __getattr__(self, name):
        if name in ['training_step', 'validation_step', 'on_train_end', 'configure_optimizers', 'model']:
            return super().__getattr__(name)
        return getattr(self.__dict__['_modules']['model'], name)
        # 我这才想起来，没有必要我定义的所有属性都转发一次。因为除了self.model之外，其他
        # 的属性到底是定义在self里面还是self.model里面，又有何分别？

    # def on_train_start(self) -> None:
    #     val_dataset = ConcatDataset([loader.dataset for loader in self.trainer.val_dataloaders])
    #     if isinstance(self.trainer.train_dataloader.loaders, list):
    #         train_dataset = ConcatDataset([loader.dataset for loader in self.trainer.train_dataloader.loaders])
    #         darts_set = DartsDataset(train_dataset, val_dataset)        
    #         darts_loader = DataLoader(darts_set, self.trainer.train_dataloader.loaders[0].batch_size)
    #     else:
    #         train_dataset = self.trainer.train_dataloader.loaders.dataset
    #         darts_set = DartsDataset(train_dataset, val_dataset)        
    #         darts_loader = DataLoader(darts_set, self.trainer.train_dataloader.loaders.batch_size)
    #     self.trainer.train_dataloader =  darts_loader
    #     self.trainer.val_dataloaders = None
    #     breakpoint()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 假设 batch 能拿到两对数据
        trn_batch, val_batch = batch
        # phase 1. architecture step
        self.ctrl_optim.zero_grad()
        if self.unrolled:
            self._unrolled_backward(trn_batch, val_batch, 2 * batch_idx)
        else:
            self._backward(val_batch, 2 * batch_idx)
        self.ctrl_optim.step()

        # phase 2: child network step
        return self.model.training_step(trn_batch, 2 * batch_idx + 1)

    def validation_step(self, *args, **kwargs):
        return

    def configure_optimizers(self):
        return self.model_optim
    
    def on_train_end(self) -> None:
        print(self.export())
        return self.model.on_train_end()
    
    def _backward(self, val_batch, batch_idx):
        """
        Simple backward with gradient descent
        """
        loss = self.model.training_step(val_batch, batch_idx)
        loss.backward()

    def _unrolled_backward(self, trn_batch, val_batch, batch_idx):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        trn_X, trn_y = trn_batch
        # do virtual step on training data
        lr = self.model_optim.param_groups[0]["lr"]
        momentum = self.model_optim.param_groups[0]["momentum"]
        weight_decay = self.model_optim.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        loss = self.model.training_step(val_batch, batch_idx)
        w_model, w_ctrl = tuple(self.model.parameters()), tuple([c.alpha for c in self.nas_modules])
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)
    

    def _compute_hessian(self, backup_params, dw, trn_X, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            warning('In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.', norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d

            loss = self.training_step((trn_X, trn_y), -1)
            dalphas.append(torch.autograd.grad(loss, [c.alpha for c in self.nas_modules]))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)
    
    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        _, loss = self._logits_and_loss(X, y)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.model_optim.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)
    
    def export(self):
        result = {}
        for name, module in self.nas_modules:
            if name not in result: # 这里有什么会导致重复的情况吗？
                result[name] = module.export()
        return result
    
class DartsTrainerPlugin(SingleDevicePlugin):
    distributed_backend = DistributedType.DP

    def __init__(self, checkpoint_io=None, unrolled=False, arc_lr = 3.e-4):
        super().__init__(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.unrolled = unrolled
        self.arc_lr = arc_lr
    
    def setup(self):
        self.nas_modules = []
        replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
        replace_input_choice(self.model, DartsInputChoice, self.nas_modules)
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), self.arc_lr, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        
        # 是直接改变了 model 对象对吧，那我其实可能不必 return？或者说是，直接 return 就行对吧。
        return self.model

    def _setup_optimizer(self, optimizer):
        # 这几个函数目前没被 call 到其实
        self.model_optim = optimizer
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # 假设 batch 能拿到两对数据
        trn_batch, val_batch = batch
        # phase 1. architecture step
        self.ctrl_optim.zero_grad()
        if self.unrolled:
            self._unrolled_backward(trn_batch, val_batch, 2 * batch_idx)
        else:
            self._backward(val_batch, 2 * batch_idx)
        self.ctrl_optim.step()

        # phase 2: child network step
        return self.model.training_step(trn_batch, 2 * batch_idx + 1)

    def _backward(self, val_batch, batch_idx):
        """
        Simple backward with gradient descent
        """
        loss = self.model.training_step(val_batch, batch_idx)
        loss.backward()

    def _unrolled_backward(self, trn_batch, val_batch, batch_idx):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        trn_X, trn_y = trn_batch
        # do virtual step on training data
        lr = self.model_optim.param_groups[0]["lr"]
        momentum = self.model_optim.param_groups[0]["momentum"]
        weight_decay = self.model_optim.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        loss = self.model.training_step(val_batch, batch_idx)
        w_model, w_ctrl = tuple(self.model.parameters()), tuple([c.alpha for c in self.nas_modules])
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)
    
    def _compute_hessian(self, backup_params, dw, trn_X, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            warning('In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.', norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += e * d

            loss = self.training_step((trn_X, trn_y), -1)
            dalphas.append(torch.autograd.grad(loss, [c.alpha for c in self.nas_modules]))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.model.parameters(), backup_params):
                param.copy_(backup)
    
    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        _, loss = self._logits_and_loss(X, y)
        gradients = torch.autograd.grad(loss, self.model.parameters())
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                m = self.model_optim.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)
    
    def on_train_end(self):
        result = {}
        for name, module in self.nas_modules:
            if name not in result: # 这里有什么会导致重复的情况吗？
                result[name] = module.export()
        print(result)