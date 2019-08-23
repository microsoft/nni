import torch
from ._nnimc_torch import TorchPruner
from ._nnimc_torch import _torch_default_get_configure, _torch_default_load_configure_file



class LevelPruner(TorchPruner):
    """Prune to an exact pruning level specification

    """
    def __init__(self, configure_list):
        """
            we suggest user to use json configure list, like [{},{}...], to set configure 
            format :
            [
                {
                    'sparsity': 0,
                    'support_type': 'default'
                },
                {
                    'sparsity': 50,
                    'support_op': conv1
                }
            ]
            if you want input multiple configure from file, you'd better use load_configure_file(path) to load 
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')


    def load_configure(self, config_path):
        config_list = _torch_default_load_configure_file(config_path, 'LevelPruner')
        for config in config_list.get('config', []):
            self.configure_list.append(config)
        
    def get_sparsity(self, configure={}):
        sparsity = configure.get('sparsity',0)
        return sparsity

    def calc_mask(self, layer_info, weight):
        sparsity = self.get_sparsity(_torch_default_get_configure(self.configure_list, layer_info))
        w_abs = weight.abs()
        k = int(weight.numel() * sparsity)
        if k == 0:
            return torch.ones(weight.shape)

        threshold = torch.topk(w_abs.view(-1), k, largest = False).values.max()
        return torch.gt(w_abs, threshold).type(weight.type())

class AGPruner(TorchPruner):
    def __init__(self, configure_list):
        """
            Configure Args
                initial_sparsity
                final_sparsity
                start_epoch
                end_epoch
                frequency
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')
        
        self.mask_list = {}
        self.now_epoch = 1

    def load_configure(self, config_path):
        config_list = _torch_default_load_configure_file(config_path, 'AGPruner')
        for config in config_list.get('config', []):
            self.configure_list.append(config)

    def compute_target_sparsity(self, now_epoch, layer_info):
        configure = _torch_default_get_configure(self.configure_list, layer_info)
        end_epoch = configure.get('end_epoch', 1)
        start_epoch = configure.get('start_epoch', 1)
        freq = configure.get('frequency', 1)
        final_sparsity = configure.get('final_sparsity', 0)
        initial_sparsity = configure.get('initial_sparsity', 0)
        if end_epoch <= start_epoch or end_epoch <= now_epoch or initial_sparsity >= final_sparsity:
            return final_sparsity
        span = ((end_epoch - start_epoch-1)//freq)*freq
        assert span > 0
        target_sparsity = (final_sparsity + 
                            (initial_sparsity - final_sparsity)*
                            (1.0 - ((now_epoch - start_epoch)/span))**3)
        return target_sparsity

    def calc_mask(self, layer_info, weight):
        now_epoch = self.now_epoch
        mask = self.mask_list.get(layer_info.name, torch.ones(weight.shape))
        target_sparsity = self.compute_target_sparsity(now_epoch, layer_info)
        k = int(weight.numel() * target_sparsity)
        if k == 0 or target_sparsity>=1 or target_sparsity<=0:
            return mask
        
        w_abs = weight.abs()*mask
        threshold = torch.topk(w_abs.view(-1), k, largest = False).values.max()
        new_mask = torch.gt(w_abs, threshold).type(weight.type())
        self.mask_list[layer_info.name] = new_mask
        return new_mask
    
    def update_epoch(self, epoch):
        if epoch <= 0:
            return
        self.now_epoch = epoch
    
    
class SensitivityPruner(TorchPruner):
    def __init__(self, configure_list):
        """
            configure Args:
                sparsity
        """
        super().__init__()
        self.configure_list = []
        if isinstance(configure_list, list):
            for configure in configure_list:
                self.configure_list.append(configure)
        else:
            raise ValueError('please init with configure list')

        self.mask_list = {}
    
    def load_configure(self, config_path):
        config_list = _torch_default_load_configure_file(config_path, 'SensitivityPruner')
        for config in config_list.get('config', []):
            self.configure_list.append(config)
        
    def get_sparsity(self, configure={}):
        sparsity = configure.get('sparsity',0)
        return sparsity
    
    def calc_mask(self, layer_info, weight):
        mask = self.mask_list.get(layer_info.name, torch.ones(weight.shape))
        weight = weight*mask
        sparsity = self.get_sparsity(_torch_default_get_configure(self.configure_list, layer_info))
        target_sparsity = sparsity * torch.std(weight).item()
        k = int(weight.numel() * target_sparsity)
        if k == 0:
            return mask
        
        w_abs = weight.abs()
        threshold = torch.topk(w_abs.view(-1), k, largest = False).values.max()
        new_mask = torch.gt(w_abs, threshold).type(weight.type())
        self.mask_list[layer_info.name] = new_mask
        return new_mask
        