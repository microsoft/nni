import torch
import math
'''
                REDISTRIBUTION
'''

def momentum_redistribution(masking, name, weight, mask):
    """Calculates momentum redistribution statistics.

    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

        mask        The binary mask. 1s indicated active weights.

    Returns:
        Layer Statistic      The unnormalized layer statistics
                    for the layer "name". A higher value indicates
                    that more pruned parameters are redistributed
                    to this layer compared to layers with lower value.
                    The values will be automatically sum-normalized
                    after this step.


    The calculation of redistribution statistics is the first
    step in this sparse learning library.
    """
    grad = masking.get_momentum_for_weight(weight)
    mean_magnitude = torch.abs(grad[mask.byte()]).mean().item()
    return mean_magnitude

def magnitude_redistribution(masking, name, weight, mask):
    mean_magnitude = torch.abs(weight)[mask.byte()].mean().item()
    return mean_magnitude

def nonzero_redistribution(masking, name, weight, mask):
    nonzero = (weight !=0.0).sum().item()
    return nonzero

def no_redistribution(masking, name, weight, mask):
    num_params = masking.baseline_nonzero
    n = weight.numel()
    return n/float(num_params)


'''
                PRUNE
'''
def magnitude_prune(masking, mask, weight, name):
    """Prunes the weights with smallest magnitude.

    The pruning functions in this sparse learning library
    work by constructing a binary mask variable "mask"
    which prevents gradient flow to weights and also
    sets the weights to zero where the binary mask is 0.
    Thus 1s in the "mask" variable indicate where the sparse
    network has active weights. In this function name
    and masking can be used to access global statistics
    about the specific layer (name) and the sparse network
    as a whole.

    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        mask        The binary mask. 1s indicated active weights.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

    Returns:
        mask        Pruned Binary mask where 1s indicated active
                    weights. Can be modified in-place or newly
                    constructed

    Accessable global statistics:

    Layer statistics:
        Non-zero count of layer:
            masking.name2nonzeros[name]
        Zero count of layer:
            masking.name2zeros[name]
        Redistribution proportion:
            masking.name2variance[name]
        Number of items removed through pruning:
            masking.name2removed[name]

    Network statistics:
        Total number of nonzero parameter in the network:
            masking.total_nonzero = 0
        Total number of zero-valued parameter in the network:
            masking.total_zero = 0
        Total number of parameters removed in pruning:
            masking.total_removed = 0
    """
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0

    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask

def global_magnitude_prune(masking):
    prune_rate = 0.0
    for name in masking.name2prune_rate:
        if name in masking.masks:
            prune_rate = masking.name2prune_rate[name]
    tokill = math.ceil(prune_rate*masking.baseline_nonzero)
    total_removed = 0
    prev_removed = 0
    while total_removed < tokill*(1.0-masking.tolerance) or (total_removed > tokill*(1.0+masking.tolerance)):
        total_removed = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue
                remain = (torch.abs(weight.data) > masking.prune_threshold).sum().item()
                total_removed += masking.name2nonzeros[name] - remain

        if prev_removed == total_removed: break
        prev_removed = total_removed
        if total_removed > tokill*(1.0+masking.tolerance):
            masking.prune_threshold *= 1.0-masking.increment
            masking.increment *= 0.99
        elif total_removed < tokill*(1.0-masking.tolerance):
            masking.prune_threshold *= 1.0+masking.increment
            masking.increment *= 0.99

    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue
            masking.masks[name][:] = torch.abs(weight.data) > masking.prune_threshold

    return int(total_removed)


def magnitude_and_negativity_prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])
    if num_remove == 0.0: return weight.data != 0.0

    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + (num_remove/2.0))

    # remove all weights which absolute value is smaller than threshold
    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0

    # remove the most negative weights
    x, idx = torch.sort(weight.data.view(-1))
    mask.data.view(-1)[idx[:math.ceil(num_remove/2.0)]] = 0.0

    return mask

'''
                GROWTH
'''

def random_growth(masking, name, new_mask, total_regrowth, weight):
    n = (new_mask==0).sum().item()
    if n == 0: return new_mask
    expeced_growth_probability = (total_regrowth/n)
    new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
    return new_mask.byte() | new_weights

def momentum_growth(masking, name, new_mask, total_regrowth, weight):
    """Grows weights in places where the momentum is largest.

    Growth function in the sparse learning library work by
    changing 0s to 1s in a binary mask which will enable
    gradient flow. Weights default value are 0 and it can
    be changed in this function. The number of parameters
    to be regrown is determined by the total_regrowth
    parameter. The masking object in conjunction with the name
    of the layer enables the access to further statistics
    and objects that allow more flexibility to implement
    custom growth functions.

    Args:
        masking     Masking class with state about current
                    layers and the entire sparse network.

        name        The name of the layer. This can be used to
                    access layer-specific statistics in the
                    masking class.

        new_mask    The binary mask. 1s indicated active weights.
                    This binary mask has already been pruned in the
                    pruning step that preceeds the growth step.

        total_regrowth    This variable determines the number of
                    parameters to regrowtn in this function.
                    It is automatically determined by the
                    redistribution function and algorithms
                    internal to the sparselearning library.

        weight      The weight of the respective sparse layer.
                    This is a torch parameter.

    Returns:
        mask        Binary mask with newly grown weights.
                    1s indicated active weights in the binary mask.

    Access to optimizer:
        masking.optimizer

    Access to momentum/Adam update:
        masking.get_momentum_for_weight(weight)

    Accessable global statistics:

    Layer statistics:
        Non-zero count of layer:
            masking.name2nonzeros[name]
        Zero count of layer:
            masking.name2zeros[name]
        Redistribution proportion:
            masking.name2variance[name]
        Number of items removed through pruning:
            masking.name2removed[name]

    Network statistics:
        Total number of nonzero parameter in the network:
            masking.total_nonzero = 0
        Total number of zero-valued parameter in the network:
            masking.total_zero = 0
        Total number of parameters removed in pruning:
            masking.total_removed = 0
    """
    grad = masking.get_momentum_for_weight(weight)
    if grad.dtype == torch.float16:
        grad = grad*(new_mask==0).half()
    else:
        grad = grad*(new_mask==0).float()
    y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
    new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

    return new_mask

def momentum_neuron_growth(masking, name, new_mask, total_regrowth, weight):
    grad = masking.get_momentum_for_weight(weight)

    M = torch.abs(grad)
    if len(M.shape) == 2: sum_dim = [1]
    elif len(M.shape) == 4: sum_dim = [1, 2, 3]

    v = M.mean(sum_dim).data
    v /= v.sum()

    slots_per_neuron = (new_mask==0).sum(sum_dim)

    M = M*(new_mask==0).float()
    for i, fraction  in enumerate(v):
        neuron_regrowth = math.floor(fraction.item()*total_regrowth)
        available = slots_per_neuron[i].item()

        y, idx = torch.sort(M[i].flatten())
        if neuron_regrowth > available:
            neuron_regrowth = available
        # TODO: Work into more stable growth method
        threshold = y[-(neuron_regrowth)].item()
        if threshold == 0.0: continue
        if neuron_regrowth < 10: continue
        new_mask[i] = new_mask[i] | (M[i] > threshold)

    return new_mask


def global_momentum_growth(masking, total_regrowth):
    togrow = total_regrowth
    total_grown = 0
    last_grown = 0
    while total_grown < togrow*(1.0-masking.tolerance) or (total_grown > togrow*(1.0+masking.tolerance)):
        total_grown = 0
        total_possible = 0
        for module in masking.modules:
            for name, weight in module.named_parameters():
                if name not in masking.masks: continue

                new_mask = masking.masks[name]
                grad = masking.get_momentum_for_weight(weight)
                grad = grad*(new_mask==0).float()
                possible = (grad !=0.0).sum().item()
                total_possible += possible
                grown = (torch.abs(grad.data) > masking.growth_threshold).sum().item()
                total_grown += grown
        if total_grown == last_grown: break
        last_grown = total_grown


        if total_grown > togrow*(1.0+masking.tolerance):
            masking.growth_threshold *= 1.02
            #masking.growth_increment *= 0.95
        elif total_grown < togrow*(1.0-masking.tolerance):
            masking.growth_threshold *= 0.98
            #masking.growth_increment *= 0.95

    total_new_nonzeros = 0
    for module in masking.modules:
        for name, weight in module.named_parameters():
            if name not in masking.masks: continue

            new_mask = masking.masks[name]
            grad = masking.get_momentum_for_weight(weight)
            grad = grad*(new_mask==0).float()
            masking.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > masking.growth_threshold)).float()
            total_new_nonzeros += new_mask.sum().item()
    return total_new_nonzeros




prune_funcs = {}
prune_funcs['magnitude'] = magnitude_prune
prune_funcs['SET'] = magnitude_and_negativity_prune
prune_funcs['global_magnitude'] = global_magnitude_prune

growth_funcs = {}
growth_funcs['random'] = random_growth
growth_funcs['momentum'] = momentum_growth
growth_funcs['momentum_neuron'] = momentum_neuron_growth
growth_funcs['global_momentum_growth'] = global_momentum_growth

redistribution_funcs = {}
redistribution_funcs['momentum'] = momentum_redistribution
redistribution_funcs['nonzero'] = nonzero_redistribution
redistribution_funcs['magnitude'] = magnitude_redistribution
redistribution_funcs['none'] = no_redistribution
