from ..operation import TensorFlowOperation

class Conv2D(TensorFlowOperation):
    def to_init_code(self, field):
        parameters = {'padding': 'same', **parameters}
        super().__init__(type, parameters, _internal_access)
