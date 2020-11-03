import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from nni.retiarii.converter.graph_gen import convert_to_graph
from nni.retiarii.converter.visualize import visualize_model
from nni.retiarii import nn
from base_mnasnet import MNASNet


if __name__ == '__main__':
    _DEFAULT_DEPTHS = [16, 24, 40, 80, 96, 192, 320]
    _DEFAULT_CONVOPS = ["dconv", "mconv", "mconv", "mconv", "mconv", "mconv", "mconv"]
    _DEFAULT_SKIPS = [False, True, True, True, True, True, True]
    _DEFAULT_KERNEL_SIZES = [3, 3, 5, 5, 3, 5, 3]
    _DEFAULT_NUM_LAYERS = [1, 3, 3, 3, 2, 4, 1]
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

    '''exp = sdk.create_experiment('mnasnet_search', base_model)
    mutators = []
    base_filter_sizes = [16, 24, 40, 80, 96, 192, 320]
    exp_ratios = [3, 3, 3, 6, 6, 6, 6]
    strides = [1, 2, 2, 2, 1, 2, 1]
    for i in range(3, 10):
        mutators.append(BlockMutator(i, 'layers__'+str(i)+'__placeholder',
                        n_layer_options=[1, 2, 3, 4],
                        op_type_options=['RegularConv', 'DepthwiseConv', 'MobileConv'],
                        kernel_size_options=[3, 5],
                        se_ratio_options=[0, 0.25],
                        #skip_options=['pool', 'identity', 'no'],
                        skip_options=['identity', 'no'],
                        n_filter_options=[int(base_filter_sizes[i-3]*x) for x in [0.75, 1.0, 1.25]],
                        exp_ratio = exp_ratios[i-3],
                        stride = strides[i-3]))
    exp.specify_training(ModelTrain)'''