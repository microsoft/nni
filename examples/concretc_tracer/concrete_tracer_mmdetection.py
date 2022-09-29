# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    
from mmdet.apis import init_detector, inference_detector
import torch
import mmcv

folder_prefix = 'C:\\Works\\mmdetection' # replace this path with yours
# Specify the path to model config and checkpoint file
config_file = '%s/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py' % folder_prefix
checkpoint_file = '%s/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' % folder_prefix

# build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
model = init_detector(config_file, device='cpu')

# # test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')



from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

def get_data(model, img):
  """Async inference image(s) with the detector.

  Args:
      model (nn.Module): The loaded detector.
      img (str | ndarray): Either image files or loaded images.

  Returns:
      Awaitable detection results.
  """
  imgs = [img]

  cfg = model.cfg
  device = next(model.parameters()).device  # model device

  cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
  test_pipeline = Compose(cfg.data.test.pipeline)

  datas = []
  for img in imgs:
    # prepare data
    # add information into dict
    data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    data = test_pipeline(data)
    datas.append(data)

  data = collate(datas, samples_per_gpu=len(imgs))
  # just get the actual data from DataContainer
  data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
  data['img'] = [img.data[0] for img in data['img']]
  if next(model.parameters()).is_cuda:
    # scatter to specified GPU
    data = scatter(data, [device])[0]
  else:
    for m in model.modules():
      assert not isinstance(
        m, RoIPool
      ), 'CPU inference with RoIPool is not supported currently.'
  return data

img = 'C:\\Works\\mmdetection\\tests\\data\\color.jpg'
# result = inference_detector(model, img)
# print('result:', result)
# model.forward = model.forward_dummy
data = get_data(model, img)
# print('data:', data)
with torch.no_grad():
  result = model(return_loss=False, rescale=True, **data)
# result = model.forward(**data)
# print('result:', result)



from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.compression.pytorch.utils import not_safe_to_prune
from nni.common.concrete_trace_utils import concrete_trace


traced_model = concrete_trace(model, {'return_loss': False, 'rescale': True, **data}, False)

out_a0 = model(**{'return_loss': False, 'rescale': True, **data})
out_a1 = traced_model(**{'return_loss': False, 'rescale': True, **data})
# print('traced code:\n', traced_model.code)

print('out_a0 == out_a1:', torch.equal(out_a0, out_a1))