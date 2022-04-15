
from turtle import Shape
from mobilenet_utils import *
from sparta.common.utils import export_tesa
from shape_hook import ShapeHook
def get_mobile_coarse():
    import torch
    from nni.compression.pytorch.speedup import ModelSpeedup
    model = create_model('mobilenet_v1')
    dummy_input = torch.rand(1,3,224,224)
    new_model = align_speedup(model, dummy_input, 'checkpoints/coarsegrained/mobilenet_0.6_align_run1/mask_temp.pth')
    state = torch.load('checkpoints/coarsegrained/mobilenet_0.6_align_run1/finetune_weights.pth')
    new_model.load_state_dict(state)
    return new_model

m = get_mobile_coarse()
device = torch.device('cuda')
test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# import ipdb; ipdb.set_trace()
# sh =  ShapeHook(m, torch.rand(32, 3, 224, 224))
# sh.export('mobilenet_coarse_shape.json')
# exit(0)
print('Accuracy:', run_eval(m.cuda(), test_dataloader, device))
print('Propagation done')

export_tesa(m.cuda(), torch.rand(32,3,224,224).cuda(), 'artifact_mobilenet_coarse_onnx_with_tesa')
