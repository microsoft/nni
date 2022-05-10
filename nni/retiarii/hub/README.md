This README will be deleted once this hub got stabilized, after which we will promote it in the documentation.

## Why

We hereby provides a series of state-of-the-art search space, which is PyTorch model + mutations + training recipe.

For further motivations and plans, please see https://github.com/microsoft/nni/issues/4249.

## Reproduction Roadmap

1. Runnable
2. Load checkpoint of searched architecture and evaluate
3. Reproduce "retrain" (i.e., training from scratch of searched architecture)
4. Runnable with built-in algos
5. Reproduce result with at least one algo

|                        | 1      | 2      | 3      | 4      | 5      |
|------------------------|--------|--------|--------|--------|--------|
| NasBench101            | Y      | -      |        |        |        |
| NasBench201            | Y      | -      |        |        |        |
| NASNet                 | Y      | -      |        |        |        |
| ENAS                   | Y      | -      |        |        |        |
| AmoebaNet              | Y      | -      |        |        |        |
| PNAS                   | Y      | -      |        |        |        |
| DARTS                  | Y      | Y      |        |        |        |
| ProxylessNAS           | Y      | Y      |        |        |        |
| MobileNetV3Space       | Y      | Y      |        |        |        |
| ShuffleNetSpace        | Y      | Y      |        |        |        |
| ShuffleNetSpace (ch)   | Y      | -      |        |        |        |

* `-`: Result unavailable, because lacking published checkpoints / architectures.
* NASNet, ENAS, AmoebaNet, PNAS, DARTS are based on the same implementation, with configuration differences.
* NasBench101 and 201 will directly proceed to stage 3 as it's cheaper to train them than to find a checkpoint.

## Space Planned

We welcome suggestions and contributions.

- [AutoFormer](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_AutoFormer_Searching_Transformers_for_Visual_Recognition_ICCV_2021_paper.html), [PR under review](https://github.com/microsoft/nni/pull/4551)
- [NAS-BERT](https://arxiv.org/abs/2105.14444)
- Something speech, like [LightSpeech](https://arxiv.org/abs/2102.04040)

## Searched Model Zoo

Create a searched model with pretrained weights like the following:

```python
model = MobileNetV3Space.load_searched_model('mobilenetv3-small-075', pretrained=True, download=True)
evaluate(model, imagenet_data)
```

``MobileNetV3Space`` can be replaced with any search space listed above, and ``mobilenetv3-small-075`` can be any model listed below.

See an example of ``evaluate`` [here](https://github.com/rwightman/pytorch-image-models/blob/d30685c283137b4b91ea43c4e595c964cd2cb6f0/train.py#L778).

| Search space     | Model                 | Dataset  | Metric | Eval Protocol                |
|------------------|-----------------------|----------|--------|------------------------------|
| ProxylessNAS     | acenas-m1             | ImageNet | 75.176 | Default                      |
| ProxylessNAS     | acenas-m2             | ImageNet | 75.0   | Default                      |
| ProxylessNAS     | acenas-m3             | ImageNet | 75.118 | Default                      |
| ProxylessNAS     | proxyless-cpu         | ImageNet | 75.29  | Default                      |
| ProxylessNAS     | proxyless-gpu         | ImageNet | 75.084 | Default                      |
| ProxylessNAS     | proxyless-mobile      | ImageNet | 74.594 | Default                      |
| MobileNetV3Space | mobilenetv3-large-100 | ImageNet | 75.768 | Bicubic interpolation        |
| MobileNetV3Space | mobilenetv3-small-050 | ImageNet | 57.906 | Bicubic interpolation        |
| MobileNetV3Space | mobilenetv3-small-075 | ImageNet | 65.24  | Bicubic interpolation        |
| MobileNetV3Space | mobilenetv3-small-100 | ImageNet | 67.652 | Bicubic interpolation        |
| MobileNetV3Space | cream-014             | ImageNet | 53.74  | Test image size = 64         |
| MobileNetV3Space | cream-043             | ImageNet | 66.256 | Test image size = 96         |
| MobileNetV3Space | cream-114             | ImageNet | 72.514 | Test image size = 160        |
| MobileNetV3Space | cream-287             | ImageNet | 77.52  | Default                      |
| MobileNetV3Space | cream-481             | ImageNet | 79.078 | Default                      |
| MobileNetV3Space | cream-604             | ImageNet | 79.92  | Default                      |
| DARTS            | darts-v2              | CIFAR-10 | 97.37  | Default                      |
| ShuffleNetSpace  | spos                  | ImageNet | 74.14  | BGR tensor; no normalization |

The metrics listed above are obtained by evaluating the checkpoints provided the original author and converted to NNI NAS format with [these scripts](https://github.com/ultmaster/spacehub-conversion). Do note that some metrics can be higher / lower than the original report, because there could be subtle differences between data preprocessing, operation implementation (e.g., 3rd-party hswish vs ``nn.Hardswish``), or even library versions we are using. But most of these errors are acceptable (~0.1%). We will retrain these architectures in a reproducible and fair training settings, and update these results when the training is ready.

Latency / FLOPs data are missing in the table. Measuring them would be another task.

Several more models to be added:

- FBNet on MobileNetV3Space
