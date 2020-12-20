from ..operation import PyTorchOperation


class relu(PyTorchOperation):
    def to_init_code(self, field):
        return ''

    def to_forward_code(self, field, output, *inputs) -> str:
        assert len(inputs) == 1
        return f'{output} = nn.functional.relu({inputs[0]})'


class Flatten(PyTorchOperation):
    def to_init_code(self, field):
        return ''

    def to_forward_code(self, field, output, *inputs) -> str:
        assert len(inputs) == 1
        return f'{output} = {inputs[0]}.view({inputs[0]}.size(0), -1)'


class ToDevice(PyTorchOperation):
    def to_init_code(self, field):
        return ''

    def to_forward_code(self, field, output, inputs) -> str:
        assert len(inputs) == 1
        return f"{output} = {inputs[0]}.to('{self.parameters['device']}')"


class Dense(PyTorchOperation):
    def to_init_code(self, field):
        return f"self.{field} = nn.Linear({self.parameters['in_features']}, {self.parameters['out_features']})"

    def to_forward_code(self, field, output, *inputs) -> str:
        assert len(inputs) == 1
        return f'{output} = self.{field}({inputs[0]})'


class Softmax(PyTorchOperation):
    def to_init_code(self, field):
        return ''

    def to_forward_code(self, field, output, *inputs) -> str:
        assert len(inputs) == 1
        return f'{output} = F.softmax({inputs[0]}, -1)'
