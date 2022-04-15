
from turtle import Shape
from mobilenet_utils import *
from sparta.common.utils import export_tesa
from shape_hook import ShapeHook

model = create_model('mobilenet_v1')
dummy_input = torch.rand(32,3,224,224)
mask = torch.load('checkpoints/coarsegrained/mobilenet_0.6_align_run1/mask_temp.pth')
export_tesa(model, dummy_input, 'artifact_mobilenet_coarsegrained_no_propagation_onnx_with_tesa', mask)