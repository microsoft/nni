from nn_meter import get_default_config, load_latency_predictors  # pylint: disable=import-error


class LatencyFilter:
    def __init__(self, threshold, config=None, hardware='', reverse=False):
        """
        Filter the models according to predcted latency.

        Parameters
        ----------
        threshold: `float`
            the threshold of latency
        config, hardware:
            determine the targeted device
        reverse: `bool`
            if reverse is `False`, then the model returns `True` when `latency < threshold`,
            else otherwisse
        """
        default_config, default_hardware = get_default_config()
        if config is None:
            config = default_config
        if not hardware:
            hardware = default_hardware

        self.predictors = load_latency_predictors(config, hardware)
        self.threshold = threshold

    def __call__(self, ir_model):
        latency = self.predictors.predict(ir_model, 'nni')
        return latency < self.threshold
