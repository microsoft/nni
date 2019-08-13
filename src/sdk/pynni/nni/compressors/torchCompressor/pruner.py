try:
    import torch
    from ._nnimc_torch import TorchPruner
    class TorchLevelPruner(TorchPruner):
        def __init__(self, sparsity = 0, layer_sparsity = { }):
            super().__init__()
            self.default_sparsity = sparsity
            self.layer_sparsity = layer_sparsity
    
        def calc_mask(self, layer_info, weight):
            sparsity = self.layer_sparsity.get(layer_info.name, self.default_sparsity)
            w_abs = weight.abs()
            k = int(weight.numel() * sparsity)
            threshold = torch.topk(w_abs.view(-1), k, largest = False).values.max()
            return torch.gt(w_abs, threshold).type(weight.type())
    
    class TorchAGPruner(TorchPruner):
        def __init__(self, initial_sparsity=0, final_sparsity=0.8, start_epoch=1, end_epoch=1, frequency=1):
            """
            """
            super().__init__()
            self.initial_sparsity = initial_sparsity
            self.final_sparsity = final_sparsity
            self.start_epoch = start_epoch
            self.end_epoch = end_epoch
            self.now_epoch = start_epoch
            self.freq = frequency
            self.mask_list = {}

        def compute_target_sparsity(self, now_epoch):
            if self.end_epoch <= self.start_epoch or self.end_epoch <= self.now_epoch:
                return self.final_sparsity
            span = ((self.end_epoch - self.start_epoch-1)//self.freq)*self.freq
            assert span>0
            target_sparsity = (self.final_sparsity + 
                                (self.initial_sparsity - self.final_sparsity)*
                                (1.0 - ((now_epoch - self.start_epoch)/span))**3 )
            return target_sparsity

        def calc_mask(self, layer_info, weight):
            now_epoch = self.now_epoch
            mask = self.mask_list.get(layer_info.name, torch.ones(weight.shape))
            target_sparsity = self.compute_target_sparsity(now_epoch)
            k = int(weight.numel() * target_sparsity)
            if k == 0:
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
        
        
    class TorchSensitivityPruner(TorchPruner):
        def __init__(self, sparsity):
            super().__init__()
            self.sparsity = sparsity
            self.mask_list = {}

        def calc_mask(self, layer_info, weight):
            mask = self.mask_list.get(layer_info.name, torch.ones(weight.shape))
            weight = weight*mask
            target_sparsity = self.sparsity * torch.std(weight).item()
            k = int(weight.numel() * target_sparsity)
            if k == 0:
                return mask
            
            w_abs = weight.abs()
            threshold = torch.topk(w_abs.view(-1), k, largest = False).values.max()
            new_mask = torch.gt(w_abs, threshold).type(weight.type())
            self.mask_list[layer_info.name] = new_mask
            return new_mask

except ModuleNotFoundError:
    pass


