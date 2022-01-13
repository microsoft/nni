from logging import  warning
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data.dataset import Dataset

def _replace_module_with_type(root_module, replace_dict, modules):
    if modules is None:
        modules = []
    def apply(m):
        for name, child in m.named_children():
            child_type = type(child)
            if child_type in replace_dict.keys():
                setattr(m, name, replace_dict[child_type](child))
                modules.append((child.key, getattr(m, name)))
            else:
                apply(child)

    apply(root_module)
    return modules

# no val, test and preditc stage for most nas methods
REDIRECT_LIGHTNING_HOOKS=[
    'on_train_start',
    'on_fit_start',
    'on_fit_end',
    'on_train_batch_start',
    'on_train_batch_end',
    'on_epoch_start',
    'on_epoch_end',
    'on_train_epoch_start',
    'on_train_epoch_end',
    'on_before_backward',
    'on_after_backward'
]


class BaseOneShotLightningModule(pl.LightningModule):
    '''
        base_model : the search space difined by user
        default_replace_dict : the replace dict used by a paticular oneshot alg
            key: type (xxxChoice) 
            val: a func that takes a xxChoice instance as input and returns an nn.Module  
        custom_replace_dict : user difined xxChoice replace functions 
    '''
    def __init__(self, base_model, default_replace_dict = None, custom_replace_dict = None):
        super().__init__()
        assert isinstance(base_model, pl.LightningModule)
        self.model = base_model

        # replace xxxChoice with respect to nas alg
        # new modules are stored in self.nas_modules
        self.nas_modules = []
        choice_replace_dict = dict(default_replace_dict)
        if custom_replace_dict is not None:
            for k,v in custom_replace_dict.items():
                assert isinstance(v, nn.Module)
                choice_replace_dict[k] = v
        _replace_module_with_type(self.model, choice_replace_dict, self.nas_modules)
    
    # redirect lightning hooks
    def __getattr__(self, name):
        if name not in REDIRECT_LIGHTNING_HOOKS:
            return super().__getattr__(name)
        return getattr(self.__dict__['_modules']['model'], name)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        # no val step for most nas methods
        return

    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
    def on_train_end(self) -> None:
        print(self.export())
        return self.model.on_train_end()
    
    # export the nas result
    def export(self):
        result = {}
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result


class MergeTrainValDataset(Dataset):
    def __init__(self, train_data = Dataset, val_data = Dataset):
        super().__init__()
        # 是不是要两个取短的？但要是他把 val_dataset 取的很短不就很可惜？
        # concat dataset
        train_dataset = train_data
        val_dataset = val_data
        self.tds = train_dataset
        self.vds = val_dataset
        if len(train_dataset) / len(val_dataset) > 1.5 or len(val_dataset) / len(train_dataset) > 1.5 :
            warning('The length difference betweeen the training dataset({}) and the validation dataset({}) ' \
                'is too large, which will result in a waste of the longer one because the oneshot nas will truncate it.'.format(
                    len(train_dataset),len(val_dataset)
                ))
    
    def __getitem__(self, index):
        return self.tds[index], self.vds[index]
    
    def __len__(self):
        return min(len(self.tds), len(self.vds))

