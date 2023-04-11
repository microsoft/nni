from assets.registry import model_zoo
from nni.common.concrete_trace_utils import concrete_trace
mod_fn = model_zoo.get('mmdet', 'cascade_rcnn')
mod = mod_fn()
concrete_trace(mod, mod.dummy_inputs)
