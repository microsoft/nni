import nni.retiarii.nn.pytorch as nn
from nni.retiarii import register_module
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.codegen import model_to_pytorch_script
from nni.retiarii.utils import get_records