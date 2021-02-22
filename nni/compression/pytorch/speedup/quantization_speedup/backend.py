# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class BaseModelSpeedup:
    """
    Base speedup class for backend engine
    """
    def __init__(self, model, config):
        """
        Parameters
        ----------
        model : pytorch model
            The model to speed up by quantization.
        config : dict
            Config recording bit number and name of layers.
        """
        self.quantize_model = model
        self.config = config

    def inference(self, input):
        """
        This function should be overrided by subclass to provide inference ability,
        which should return output and inference time.

        Parameters
        ----------
        input : numpy data
            input data given to the inference engine

        Returns
        -------
        output : numpy data
            output data will be generated after inference
        latency : float
            latency of such inference process
        """
        raise NotImplementedError('Backend engine must overload inference()')

    def compress(self):
        """
        This function should be overrided by subclass to build inference
        engine which will be used to process input data
        """
        raise NotImplementedError('Backend engine must overload build()')

    def export_quantized_model(self, path):
        """
        This function should be overrided by subclass to build inference
        engine which will be used to process input data
        """
        raise NotImplementedError('Backend engine must overload export_engine()')