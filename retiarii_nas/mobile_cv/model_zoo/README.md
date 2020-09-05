# FBNet Model Zoo

We provide a set of efficient pre-trained pytorch models for FBNet/FBNetV2. We also provide *quantized* FBNet models in TorchScript format.

## Model Zoo Usage

Here is an example code to create a pretrained FBNet model:

```python
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from mobile_cv.model_zoo.models.preprocess import get_preprocess

model_name = "dmasking_l3"
model = fbnet(model_name, pretrained=True)
model.eval()
# get preprocess function
preprocess = get_preprocess(model.arch_def.get("input_size", 224))
```

The input image requires the following preprocessing:
```python
# load and process input
# `input_inage` in RGB format in the range of [0..255]
input_image = _get_input()
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
```

The model could be run on the image as follow:
```python
with torch.no_grad():
    output = model(input_batch)
output_softmax = torch.nn.functional.softmax(output[0], dim=0)
print(output_softmax.max(0))
```

A full example code is available [here](../../examples/run_fbnet_v2.py).

## Pretrained Models

The following FBNet/FBNetV2 pre-trained models are provided. The models are trained and evaluated using ImageNet 1k ([ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)) dataset. Validation top-1 and top-5 accuracy for fp32 models are reported.

|     Model      | Resolution | Flops (M) | Params (M) | Top-1 Accuracy | Top-5 Accuracy |
| -------------- | ---------- | --------- | ---------- | -------------- | -------------- |
| fbnet_a        | 224x224    | 244.5     | 4.3        | 73.3           | 90.9           |
| fbnet_b        | 224x224    | 291.1     | 4.8        | 74.5           | 91.8           |
| fbnet_c        | 224x224    | 378.2     | 5.5        | 75.2           | 92.3           |
| dmasking_f1    | 128x128    | 56.3      | 6.0        | 68.3           | 87.8           |
| dmasking_f4    | 224x224    | 235.9     | 7.0        | 75.5           | 92.5           |
| dmasking_l2_hs | 256x256    | 419.1     | 8.4        | 77.7           | 93.7           |
| dmasking_l3    | 288x288    | 753.1     | 9.4        | 78.9           | 94.3           |


## Quantized Models

The following FBNet quantized pre-trained models are provided. The models are trained and evaluated using ImageNet 1k ([ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)) dataset. Validation top-1 and top-5 accuracy for the int8 models are reported.

|        Model         | Resolution | Flops (M) | Params (M) | Top-1 Accuracy | Top-5 Accuracy |
| -------------------- | ---------- | --------- | ---------- | -------------- | -------------- |
| fbnet_a_i8f_int8_jit | 224x224    | 244.5     | 4.3        | 72.2           | 90.3           |
| fbnet_b_i8f_int8_jit | 224x224    | 291.1     | 4.8        | 73.2           | 91.1           |
| fbnet_c_i8f_int8_jit | 224x224    | 378.2     | 5.5        | 74.2           | 91.8           |

The quantized models could be loaded in a similar way:
```python
from mobile_cv.model_zoo.models.model_jit import model_jit
from mobile_cv.model_zoo.models.preprocess import get_preprocess
# the model is quantized with qnnpack backend
torch.backends.quantized.engine = "qnnpack"

model_name = "fbnet_c_i8f_int8_jit"
model = model_jit(model_name)
model.eval()
# get preprocess function
preprocess = get_preprocess(224)
```

A full example code is available [here](../../examples/run_fbnet_v2_jit_int8.py).
