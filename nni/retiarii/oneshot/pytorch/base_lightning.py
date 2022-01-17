from logging import  warning
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

class BaseOneShotLightningModule(pl.LightningModule):
    ''' The base class for all one-shot nas models.

    Args:
        base_model (LightningModule) : the model with xxxChoice provided by user
        custom_replace_dict (dict) : user difined xxChoice replace functions
            key (type) : the xxx choice to be replace
            val (function) : the type to be replaced with, should return a nn.Module
    '''
    def __init__(self, base_model, custom_replace_dict = None):
        super().__init__()
        assert isinstance(base_model, pl.LightningModule)
        self.model = base_model

        # replace xxxChoice with respect to nas alg
        # replaced modules are stored in self.nas_modules
        self.nas_modules = []
        choice_replace_dict = self.default_replace_dict
        if custom_replace_dict is not None:
            for k,v in custom_replace_dict.items():
                assert isinstance(v, nn.Module)
                choice_replace_dict[k] = v
        _replace_module_with_type(self.model, choice_replace_dict, self.nas_modules)
    
    def __getattr__(self, name):
        ''' redirect lightning hooks. 
            only hooks in the list below will be redirect to user-deifend ones.
            ** Note that validation related hooks are bypassed as default. **
        '''
        if name in [
            'on_train_end', 'on_fit_start', 'on_fit_end',
            'on_train_batch_start', 'on_train_batch_end', 
            'on_epoch_start', 'on_epoch_end', 
            'on_train_epoch_start', 'on_train_epoch_end', 
            'on_before_backward', 'on_after_backward'
        ]:
            return getattr(self.__dict__['_modules']['model'], name)    
        return super().__getattr__(name)
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        print('Please Note that validation step is skipped by the nas method you chose.')

    def configure_optimizers(self):
        # Combine arc optimizers and user's model optimizers
        # Overwrite configure_architecture_optimizers if optimizers are needed
        #   for your customized NAS algorithm
        arc_optimizers = self.configure_architecture_optimizers()
        if arc_optimizers is None:
            return self.model.configure_optimizers()
        
        if isinstance(arc_optimizers, optim.Optimizer):
            arc_optimizers = [arc_optimizers]
        self.arc_optim_count = len(arc_optimizers)

        w_optimizers = self.model.configure_optimizers()
        if isinstance(w_optimizers, optim.Optimizer):
            w_optimizers = [w_optimizers]
        else:
            w_optimizers = list(w_optimizers)
            
        return arc_optimizers  + w_optimizers
        
    def configure_architecture_optimizers(self):
        return None
    
    @property
    def default_replace_dict(self):
        return {}
    
    def export(self):
        # Export the Nas result, each nas module.
        # You may implement a nexport method for your customized nas_module
        result = {}
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result


class ParallelTrainValDataset(Dataset):
    ''' Dataset used by NAS algorithms. Some NAS algs may require both train_batch
        and val_bath at the training_step. 
        In this implement, the shorter one is upsampled to match the longer one.
    '''
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_dataset = train_data.dataset if isinstance(train_data, DataLoader) else train_data
        self.val_dataset = val_data.dataset if isinstance(val_data, DataLoader) else val_data
        if len(self.train_dataset) / len(self.val_dataset) > 1.5 or len(self.val_dataset) / len(self.train_dataset) > 1.5 :
            warning(f'The length difference betweeen the training dataset({len(self.train_dataset)})' \
                f' and the validation dataset({len(self.val_dataset)}) is too large. The shorter one will be upsampled.')
    
    def __getitem__(self, index):
        # The shorter one is upsampled
        return self.train_dataset[index % len(self.train_dataset)], self.val_dataset[index % len(self.val_dataset)]
    
    def __len__(self):
        return max(len(self.train_dataset), len(self.val_dataset))

class ConcatenateTrainValDataset(Dataset):
    ''' Dataset used by NAS algorithms. Some NAS algs may require both train_batch
        and val_bath at the training_step. 
        In this implement, the validation batches come after the train ones.
    '''
    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_dataset = train_data if isinstance(train_data, Dataset) else train_data.dataset
        self.val_dataset = val_data if isinstance(val_data, Dataset) else val_data.dataset
        
    def __getitem__(self, index):
        if index < len(self.train_dataset):
            return (self.train_dataset[index], True)
        return (self.val_dataset[index - len(self.train_dataset)], False)
    
    def __len__(self):
        return len(self.train_dataset)+ len(self.val_dataset)

class ParallelTrainValDataLoader(DataLoader):
    def __init__(self, train_dataloader, val_dataloader):
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.equal_len = len(train_dataloader) == len(val_dataloader)
        self.train_longger = len(train_dataloader) > len(val_dataloader)
        super().__init__(None)
    
    def __iter__(self):
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)
        return self
    
    def __next__(self):
        try:
            train_batch = next(self.train_iter)
        except StopIteration:
            if self.equal_len or self.train_longger:
                raise StopIteration()
            # val is the longger one
            self.train_iter = iter(self.train_loader)
            train_batch = next(self.train_iter)
        try:
            val_batch = next(self.val_iter)
        except StopIteration:
            if not self.train_longger:
                raise StopIteration()
            self.val_iter = iter(self.val_loader)
            val_batch = next(self.val_iter)
        return train_batch, val_batch
    
    def __len__(self) -> int:
        return max(len(self.train_loader), len(self.val_loader))



class ParallelTrainValDataLoader(DataLoader):
    def __init__(self, train_data, val_data, batch_size = 1, shuffle = False):
        if isinstance(train_data, DataLoader):
            train_data = train_data.dataset
        if isinstance(val_data, DataLoader):
            val_data = val_data.dataset
        merged_dataset = ParallelTrainValDataset(train_data, val_data)
        super().__init__(merged_dataset, batch_size, shuffle)

class ConcatenateTrainValDataLoader(DataLoader):
    def __init__(self, train_dataloader, val_dataloader):
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        super().__init__(None)
    
    def __iter__(self):
        self.cur_iter = iter(self.train_loader)
        self.is_train = True
        return self
    
    def __next__(self):
        try:
            batch = next(self.cur_iter)
        except StopIteration:
            if self.is_train:
                self.cur_iter = iter(self.val_loader)
                self.is_train = False
                return next(self)
            raise StopIteration()
        else:
            return batch, self.is_train

    def __len__(self):
        return len(self.train_loader) + len(self.val_loader)