
from mobilenet_utils import *
from nni.compression.pytorch.speedup import ModelSpeedup
from sparta.common.utils import export_tesa

model = create_model('mobilenet_v1').cuda()
dummy_input = torch.rand(32,3,224,224).cuda()

mask_path = 'checkpoints/finegrained/mobilenet_0.95_finegrain_run2/mask_temp.pth'
mask = torch.load(mask_path)
export_tesa(model, dummy_input, 'artifact_mobilenet_finegrained_no_propagation', mask)