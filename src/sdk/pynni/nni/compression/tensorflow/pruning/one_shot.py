import tensorflow as tf

from ..compressor import Pruner

__all__ = [
    'OneshotPruner'
]

class OneshotPruner(Pruner):
    def __init__(self, model, config_list, pruning_algorithm='level', optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, optimizer)
        self.set_wrapper_attribute('if_calculated', False)
        self.masker = MASKER_DICT[pruning_alogrithm](model, self, **algo_kwargs)

    def validate_config(self, model, config_list):
        pass

    def calc_mask(self, wrapper, wrapper_idx=None):
        if wrapper.if_calculated:
            return None

        sparsity = wrapper.config['sparsity']
        if not wrapper.if_calculated:
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)

            if masks is not None:
                wrapper.if_calculated = True
            return masks
        else:
            return None


MASKER_DICT = {
    'level': LevelPrunerMasker,
}

class WeightMasker:
    def __init__(self, model, pruner, **kwargs):
        self.model = model
        self.pruner = pruner

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        raise NotImplementedError()

class LevelPrunerMasker(WeightMasker):
    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        weight = wrapper.module.weight * wrapper.weight_mask
        w_abs = tf.abs(w_abs)
        k = int(tf.size(weight) * sparsity)
        assert k > 0  # FIXME
        threshold = tf.reduce_max(tf.topk(tf.reshape(w_abs, [-1]), k, largest=False)[0])
        mask_weight = tf.cast((w_abs > threshold), weight.dtype)
        return {'weight_mask': mask_weight}
