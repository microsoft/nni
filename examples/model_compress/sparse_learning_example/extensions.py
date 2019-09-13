import math
import torch

# Through the masking variable we have access to the following variables/statistics.
'''
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
'''

def your_redistribution(masking, name, weight, mask):
    '''
    Returns:
        Layer importance      The unnormalized layer importance statistic
                    for the layer "name". A higher value indicates
                    that more pruned parameters are redistributed
                    to this layer compared to layers with lower value.
                    The values will be automatically sum-normalized
                    after this step.
    '''
    return layer_importance

#===========================================================#
#                         EXAMPLE                           #
#===========================================================#
def variance_redistribution(masking, name, weight, mask):
    '''Return the mean variance of existing weights.

    Higher gradient variance means a layer does not have enough
    capacity to model the inputs with the current number of weights.
    Thus we want to add more weights if we have higher variance.
    If variance of the gradient stabilizes this means
    that some weights might be useless/not needed.
    '''
    # Adam calculates the running average of the sum of square for us
    # This is similar to RMSProp. 
    if 'exp_avg_sq' not in masking.optimizer.state[weight]:
        print('Variance redistribution requires the adam optimizer to be run!')
        raise Exception('Variance redistribution requires the adam optimizer to be run!')
    iv_adam_sumsq = torch.sqrt(masking.optimizer.state[weight]['exp_avg_sq'])

    layer_importance = iv_adam_sumsq[mask.byte()].mean().item()
    return layer_importance


def your_pruning(masking, mask, weight, name):
    """Returns:
        mask        Pruned Binary mask where 1s indicated active
                    weights. Can be modified in-place or newly
                    constructed
    """
    return mask

#===========================================================#
#                         EXAMPLE                           #
#===========================================================#
def magnitude_variance_pruning(masking, mask, weight, name):
    ''' Prunes weights which have high gradient variance and low magnitude.

    Intuition: Weights that are large are important but there is also a dimension
    of reliability. If a large weight makes a large correct prediction 8/10 times
    is it better than a medium weight which makes a correct prediction 10/10 times?
    To test this, we combine magnitude (importance) with reliability (variance of
    gradient).

    Good:
        Weights with large magnitude and low gradient variance are the most important.
        Weights with medium variance/magnitude are promising for improving network performance.
    Bad:
        Weights with large magnitude but high gradient variance hurt performance.
        Weights with small magnitude and low gradient variance are useless.
        Weights with small magnitude and high gradient variance cannot learn anything usefull.

    We here take the geometric mean of those both normalized distribution to find weights to prune.
    '''
    # Adam calculates the running average of the sum of square for us
    # This is similar to RMSProp. We take the inverse of this to rank
    # low variance gradients higher.
    if 'exp_avg_sq' not in masking.optimizer.state[weight]:
        print('Magnitude variance pruning requires the adam optimizer to be run!')
        raise Exception('Magnitude variance pruning requires the adam optimizer to be run!')
    iv_adam_sumsq = 1./torch.sqrt(masking.optimizer.state[weight]['exp_avg_sq'])

    num_remove = math.ceil(masking.name2prune_rate[name]*masking.name2nonzeros[name])

    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0: return weight.data != 0.0

    max_var = iv_adam_sumsq[mask.byte()].max().item()
    max_magnitude = torch.abs(weight.data[mask.byte()]).max().item()
    product = ((iv_adam_sumsq/max_var)*torch.abs(weight.data)/max_magnitude)*mask
    product[mask==0] = 0.0

    x, idx = torch.sort(product.view(-1))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask


def your_growth(masking, name, new_mask, total_regrowth, weight):
    '''
    Returns:
        mask        Binary mask with newly grown weights.
                    1s indicated active weights in the binary mask.
    '''
    return new_mask


