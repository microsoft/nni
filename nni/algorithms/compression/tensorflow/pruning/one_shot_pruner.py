import tensorflow as tf

from nni.compression.tensorflow import Pruner

__all__ = [
    'LevelPruner',
    'SlimPruner',
]

class OneshotPruner(Pruner):
    def __init__(self, model, config_list, masker_class, **algo_kwargs):
        super().__init__(model, config_list)
        self.set_wrappers_attribute('calculated', False)
        self.masker = masker_class(model, self, **algo_kwargs)

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
        super().__init__(model, config_list, LevelPrunerMasker)


class SlimPruner(OneshotPruner):
    def __init__(self, model, config_list):
        super().__init__(model, config_list, SlimPrunerMasker)


class WeightMasker:
    def __init__(self, model, pruner, **kwargs):
        self.model = model
        self.pruner = pruner

    def calc_masks(self, sparsity, wrapper, wrapper_idx=None):
        raise NotImplementedError()


class LevelPrunerMasker(WeightMasker):
    def calc_masks(self, sparsity, wrapper, wrapper_idx=None):
        masks = {}
        for weight_variable in wrapper.layer.weights:
            if 'bias' in weight_variable.name:
                continue

            num_prune = int(tf.size(weight_variable).numpy() * sparsity)
            if num_prune == 0:
                continue

            weight = weight_variable.read_value()
            if wrapper.masks.get(weight_variable.name) is not None:
                weight = tf.math.multiply(weight, wrapper.masks[weight_variable.name])

            w_abs = tf.math.abs(weight)
            k = tf.size(weight) - num_prune
            topk = tf.math.top_k(tf.reshape(w_abs, [-1]), k).values
            if tf.size(topk) == 0:
                mask = tf.zeros_like(weight)
            else:
                mask = tf.math.greater_equal(w_abs, topk[-1])
            masks[weight_variable.name] = tf.cast(mask, weight.dtype)
        return masks

class SlimPrunerMasker(WeightMasker):
    def __init__(self, model, pruner, **kwargs):
        super().__init__(model, pruner)
        weight_list = []
        for wrapper in pruner.wrappers:
            weights = [w for w in wrapper.layer.weights if '/gamma:' in w.name]
            assert len(weights) == 1, f'Bad weights: {[w.name for w in wrapper.layer.weights]}'
            weight_list.append(tf.math.abs(weights[0].read_value()))
        all_bn_weights = tf.concat(weight_list, 0)
        k = int(all_bn_weights.shape[0] * pruner.wrappers[0].config['sparsity'])
        top_k = -tf.math.top_k(-tf.reshape(all_bn_weights, [-1]), k).values
        self.global_threshold = top_k.numpy()[-1]

    def calc_masks(self, sparsity, wrapper, wrapper_idx=None):
        assert isinstance(wrapper.layer, tf.keras.layers.BatchNormalization), \
                'SlimPruner only supports 2D batch normalization layer pruning'

        weight = None
        weight_name = None
        bias_name = None

        for variable in wrapper.layer.weights:
            if '/gamma:' in variable.name:
                weight = variable.read_value()
                weight_name = variable.name
            elif '/beta:' in variable.name:
                bias_name = variable.name

        assert weight is not None
        if wrapper.masks.get(weight_name) is not None:
            weight *= wrapper.masks[weight_name]

        mask = tf.cast(tf.math.abs(weight) > self.global_threshold, weight.dtype)

        masks = {weight_name: mask}
        if bias_name:
            masks[bias_name] = mask
        return masks
