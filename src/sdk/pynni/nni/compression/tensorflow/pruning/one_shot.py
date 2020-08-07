import tensorflow as tf

from ..compressor import Pruner

__all__ = [
    'OneshotPruner',
    'LevelPruner',
]

class OneshotPruner(Pruner):
    def __init__(self, model, config_list, pruning_algorithm='level', **algo_kwargs):
        super().__init__(model, config_list)
        self.set_wrappers_attribute('calculated', False)
        self.masker = MASKER_DICT[pruning_algorithm](model, self, **algo_kwargs)

    def validate_config(self, model, config_list):
        pass  # TODO

    def calc_masks(self, wrapper, wrapper_idx=None):
        if wrapper.calculated:
            return None
        sparsity = wrapper.config['sparsity']
        masks = self.masker.calc_masks(sparsity, wrapper, wrapper_idx)
        if masks is not None:
            wrapper.calculated = True
        return masks


class LevelPruner(OneshotPruner):
    def __init__(self, model, config_list):
        super().__init__(model, config_list, pruning_algorithm='level')


class WeightMasker:
    def __init__(self, model, pruner, **kwargs):
        self.model = model
        self.pruner = pruner

    def calc_masks(self, sparsity, wrapper, wrapper_idx=None):
        raise NotImplementedError()


class LevelPrunerMasker(WeightMasker):
    def calc_masks(self, sparsity, wrapper, wrapper_idx=None):
        masks = {}
        for i, weight_variable in enumerate(wrapper.layer.weights):
            if weight_variable.name == 'bias':
                continue

            k = int(tf.size(weight_variable).numpy() * sparsity)
            if k == 0:
                continue

            weight = weight_variable.read_value()
            if wrapper.masks.get(weight_variable.name) is not None:
                weight = tf.math.multiply(weight, wrapper.masks[weight_variable.name])

            w_abs = tf.math.abs(tf.reshape(weight, [-1]))
            threshold = tf.math.top_k(w_abs, k)[0][0]
            mask = tf.math.greater(w_abs, threshold)
            masks[weight_variable.name] = tf.cast(mask, weight.dtype)
        return masks


MASKER_DICT = {
    'level': LevelPrunerMasker,
}
