from ..operation import TensorFlowOperation


class Conv2D(TensorFlowOperation):
    def __init__(self, type_name, parameters, _internal):
        if 'padding' not in parameters:
            parameters['padding'] = 'same'
        super().__init__(type_name, parameters, _internal)
