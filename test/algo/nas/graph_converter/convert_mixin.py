import torch

from nni.retiarii.converter.graph_gen import convert_to_graph, GraphConverterWithShape


class ConvertMixin:
    @staticmethod
    def _convert_model(model, input):
        script_module = torch.jit.script(model)
        model_ir = convert_to_graph(script_module, model)
        return model_ir


class ConvertWithShapeMixin:
    @staticmethod
    def _convert_model(model, input):
        script_module = torch.jit.script(model)
        model_ir = convert_to_graph(script_module, model, converter=GraphConverterWithShape(), dummy_input=input)
        return model_ir
