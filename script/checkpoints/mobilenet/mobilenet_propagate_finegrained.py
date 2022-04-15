
from mobilenet_utils import *
from nni.compression.pytorch.speedup import ModelSpeedup
from sparta.common.utils import export_tesa

model = create_model('mobilenet_v1').cuda()
dummy_input = torch.rand(32,3,224,224).cuda()

mask_path = 'checkpoints/finegrained/mobilenet_0.95_finegrain_run2/mask_temp.pth'
mask = torch.load(mask_path)
weight =  torch.load('checkpoints/finegrained/mobilenet_0.95_finegrain_run2/finetune_weights.pth')
ms = ModelSpeedup(model, dummy_input, mask_path)
ms.speedup_model()
model.load_state_dict(weight)
# import pdb; pdb.set_trace()
# ms =  ModelSpeedup()
device = torch.device('cuda')
test_dataset = EvalDataset('./data/stanford-dogs/Processed/test')
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# apply_mask(model, mask)
# import ipdb; ipdb.set_trace()
print('Accuracy: ',run_eval(model.cuda(), test_dataloader, device))
print('Propagation done')
# import pdb; pdb.set_trace()
export_tesa(model.cuda(), dummy_input, 'artifact_mobilenet_finegrained_onnx_with_tesa')