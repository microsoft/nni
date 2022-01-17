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

