import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from nni.retiarii.converter.graph_gen import convert_to_graph
from nni.retiarii.converter.visualize import visualize_model
from nni.retiarii import nn
from mnasnet import MNASNet


_DEFAULT_DEPTHS = [16, 24, 40, 80, 96, 192, 320]
_DEFAULT_CONVOPS = ["dconv", "mconv", "mconv", "mconv", "mconv", "mconv", "mconv"]
_DEFAULT_SKIPS = [False, True, True, True, True, True, True]
_DEFAULT_KERNEL_SIZES = [3, 3, 5, 5, 3, 5, 3]
_DEFAULT_NUM_LAYERS = [1, 3, 3, 3, 2, 4, 1]


if __name__ == '__main__':
    nn.enable_record_args()
    base_model = MNASNet(0.5, _DEFAULT_DEPTHS, _DEFAULT_CONVOPS, _DEFAULT_KERNEL_SIZES,
                    _DEFAULT_NUM_LAYERS, _DEFAULT_SKIPS)
    recorded_module_args = nn.get_records()
    nn.disable_record_args()
    print(recorded_module_args)
    script_module = torch.jit.script(base_model)
    model = convert_to_graph(script_module, base_model, recorded_module_args)
    print("Model: ", model)
    graph_ir = model._dump()
    print(graph_ir)
    visualize_model(graph_ir)
