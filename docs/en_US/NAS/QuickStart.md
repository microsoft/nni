# NAS Quick Start

The NAS feature provided by NNI has two key components: APIs for expressing the search space and NAS training approaches. The former is for users to easily specify a class of models (i.e., the candidate models specified by the search space) which may perform well. The latter is for users to easily apply state-of-the-art NAS training approaches on their own model.

Here we use a simple example to demonstrate how to tune your model architecture with the NNI NAS APIs step by step. The complete code of this example can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/naive).

## Write your model with NAS APIs

Instead of writing a concrete neural model, you can write a class of neural models using two of the NAS APIs library functions, `LayerChoice` and `InputChoice`. For example, if you think either of two options might work in the first convolution layer, then you can get one from them using `LayerChoice` as shown by `self.conv1` in the code. Similarly, the second convolution layer `self.conv2` also chooses one from two options. To this line, four candidate neural networks are specified. `self.skipconnect` uses `InputChoice` to specify two choices, adding a skip connection or not.

```python
import torch.nn as nn
from nni.nas.pytorch.mutables import LayerChoice, InputChoice

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = LayerChoice([nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(3, 6, 5, padding=2)])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = LayerChoice([nn.Conv2d(6, 16, 3, padding=1), nn.Conv2d(6, 16, 5, padding=2)])
        self.conv3 = nn.Conv2d(16, 16, 1)

        self.skipconnect = InputChoice(n_candidates=1)
        self.bn = nn.BatchNorm2d(16)

        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

For a detailed description of `LayerChoice` and `InputChoice`, please refer to [the NAS guide](NasGuide.md)

## Choose a NAS trainer

After the model is instantiated, it is time to train the model using a NAS trainer. Different trainers use different approaches to search for the best one from a class of neural models that you specified. NNI provides several popular NAS training approaches such as DARTS and ENAS. Here we use `DartsTrainer` in the example below. After the trainer is instantiated, invoke `trainer.train()` to do the search.

```python
trainer = DartsTrainer(net,
                       loss=criterion,
                       metrics=accuracy,
                       optimizer=optimizer,
                       num_epochs=2,
                       dataset_train=dataset_train,
                       dataset_valid=dataset_valid,
                       batch_size=64,
                       log_frequency=10)
trainer.train()
```

## Export the best model

After the search (i.e., `trainer.train()`) is done, to get the best performing model we simply call `trainer.export("final_arch.json")` to export the found neural architecture to a file.

## NAS visualization

We are working on NAS visualization and will release this feature soon.

## Retrain the exported best model

It is simple to retrain the found (exported) neural architecture. Step one, instantiate the model you defined above. Step two, invoke `apply_fixed_architecture` to the model. Then the model becomes the found (exported) one. Afterward, you can use traditional training to train this model.

```python
model = Net()
apply_fixed_architecture(model, "final_arch.json")
```
