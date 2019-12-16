# NNI NAS Programming Interface

We are trying to support various NAS algorithms with unified programming interface, and it's still in experimental stage. It means the current programing interface might be updated in future.

## Programming interface for user model

The programming interface of designing and searching a model is often demanded in two scenarios.

1. When designing a neural network, there may be multiple operation choices on a layer, sub-model, or connection, and it's undetermined which one or combination performs  best. So, it needs an easy way to express the candidate layers or sub-models.
2. When applying NAS on a neural network, it needs an unified way to express the search space of architectures, so that it doesn't need to update trial code for different searching algorithms.


For expressing neural architecture search space in user code, we provide the following APIs (take PyTorch as example):

```python
# in PyTorch module class
def __init__(self):
    ...
    # choose one ``op`` from ``ops``, for PyTorch this is a module.
    # op_candidates: for PyTorch ``ops`` is a list of modules, for tensorflow it is a list of keras layers.
    # key: the name of this ``LayerChoice`` instance
    self.one_layer = nni.nas.pytorch.LayerChoice([
        PoolBN('max', channels, 3, stride, 1, affine=False),
        PoolBN('avg', channels, 3, stride, 1, affine=False),
        FactorizedReduce(channels, channels, affine=False),
        SepConv(channels, channels, 3, stride, 1, affine=False),
        DilConv(channels, channels, 3, stride, 2, 2, affine=False)],
        key="layer_name")
    ...

def forward(self, x):
    ...
    out = self.one_layer(x)
    ...
```
This is for users to specify multiple candidate operations for a layer, one operation will be chosen at last. `key` is the identifier of the layer,it could be used to share choice between multiple `LayerChoice`. For example, there are two `LayerChoice` with the same candidate operations, and you want them to have the same choice (i.e., if first one chooses the `i`th op, the second one also chooses the `i`th op), give them the same key.

```python
def __init__(self):
    ...
    # choose ``n_selected`` from ``n_candidates`` inputs.
    # n_candidates: the number of candidate inputs
    # n_chosen: the number of chosen inputs
    # key: the name of this ``InputChoice`` instance
    self.input_switch = nni.nas.pytorch.InputChoice(
        n_candidates=3,
        n_chosen=1,
        key="switch_name")
    ...

def forward(self, x):
    ...
    out = self.input_switch([in_tensor1, in_tensor2, in_tensor3])
    ...
```
`InputChoice` is a PyTorch module, in init, it needs meta information, for example, from how many input candidates to choose how many inputs, and the name of this initialized `InputChoice`. The real candidate input tensors can only be obtained in `forward` function. In the `forward` function, the `InputChoice` module you create in `__init__` (e.g., `self.input_switch`) is called with real candidate input tensors.

Some [NAS trainers](#one-shot-training-mode) need to know the source layer the input tensors, thus, we add one input argument `choose_from` in `InputChoice` to indicate the source layer of each candidate input. `choose_from` is a list of string, each element is `key` of `LayerChoice` and `InputChoice` or the name of a module (refer to [the code](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/nas/pytorch/mutables.py) for more details).


Besides `LayerChoice` and `InputChoice`, we also provide `MutableScope` which allows users to label a sub-network, thus, could provide more semantic information (e.g., the structure of the network) to NAS trainers. Here is an example:
```python
class Cell(mutables.MutableScope):
    def __init__(self, scope_name):
        super().__init__(scope_name)
        self.layer1 = nni.nas.pytorch.LayerChoice(...)
        self.layer2 = nni.nas.pytorch.LayerChoice(...)
        self.layer3 = nni.nas.pytorch.LayerChoice(...)
        ...
```
The three `LayerChoice` (`layer1`, `layer2`, `layer3`) are included in the `MutableScope` named `scope_name`. NAS trainer could get this hierarchical structure.


## Two training modes

After writing your model with search space embedded in the model using the above APIs, the next step is finding the best model from the search space. There are two training modes: [one-shot training mode](#one-shot-training-mode) and [classic distributed search](#classic-distributed-search).

### One-shot training mode

Similar to optimizers of deep learning models, the procedure of finding the best model from search space can be viewed as a type of optimizing process, we call it `NAS trainer`. There have been several NAS trainers, for example, `DartsTrainer` which uses SGD to train architecture weights and model weights iteratively, `ENASTrainer` which uses a controller to train the model. New and more efficient NAS trainers keep emerging in research community.

NNI provides some popular NAS trainers, to use a NAS trainer, users could initialize a trainer after the model is defined:

```python
# create a DartsTrainer
trainer = DartsTrainer(model,
                       loss=criterion,
                       metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                       optimizer=optim,
                       num_epochs=args.epochs,
                       dataset_train=dataset_train,
                       dataset_valid=dataset_valid,)
# finding the best model from search space
trainer.train()
# export the best found model
trainer.export(file='./chosen_arch')
```

Different trainers could have different input arguments depending on their algorithms. Please refer to [each trainer's code](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch) for detailed arguments. After training, users could export the best one of the found models through `trainer.export()`. No need to start an NNI experiment through `nnictl`.

The supported trainers can be found [here](Overview.md#supported-one-shot-nas-algorithms). A very simple example using NNI NAS API can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/simple/train.py).

### Classic distributed search

Neural architecture search is originally executed by running each child model independently as a trial job. We also support this searching approach, and it naturally fits in NNI hyper-parameter tuning framework, where tuner generates child model for next trial and trials run in training service.

For using this mode, no need to change the search space expressed with NNI NAS API (i.e., `LayerChoice`, `InputChoice`, `MutableScope`). After the model is initialized, apply the function `get_and_apply_next_architecture` on the model. One-shot NAS trainers are not used in this mode. Here is a simple example:
```python
class Net(nn.Module):
    # defined model with LayerChoice and InputChoice
    ...

model = Net()
# get the chosen architecture from tuner and apply it on model
get_and_apply_next_architecture(model)
# your code for training the model
train(model)
# test the trained model
acc = test(model)
# report the performance of the chosen architecture
nni.report_final_result(acc)
```

The search space should be automatically generated and sent to tuner. As with NNI NAS API the search space is embedded in user code, users could use "[nnictl ss_gen](../Tutorial/Nnictl.md)" to generate search space file. Then, put the path of the generated search space in the field `searchSpacePath` of `config.yml`. The other fields in `config.yml` can be filled by referring [this tutorial](../Tutorial/QuickStart.md).

You could use [NNI tuners](../Tuner/BuiltinTuner.md) to do the search.

We support standalone mode for easy debugging, where you could directly run the trial command without launching an NNI experiment. This is for checking whether your trial code can correctly run. The first candidate(s) are chosen for `LayerChoice` and `InputChoice` in this standalone mode.

The complete example code can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/classic_nas/config_nas.yml).

## Programming interface for NAS algorithm

We also provide simple interface for users to easily implement a new NAS trainer on NNI.

### Implement a new NAS trainer on NNI

To implement a new NAS trainer, users basically only need to implement two classes by inheriting `BaseMutator` and `BaseTrainer` respectively.

In `BaseMutator`, users need to overwrite `on_forward_layer_choice` and `on_forward_input_choice`, which are the implementation of `LayerChoice` and `InputChoice` respectively. Users could use property `mutables` to get all `LayerChoice` and `InputChoice` in the model code. Then users need to implement a new trainer, which instantiates the new mutator and implement the training logic. For details, please read [the code](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch) and the supported trainers, for example, [DartsTrainer](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch/darts).

### Implement an NNI tuner for NAS

NNI tuner for NAS takes the auto generated search space. The search space format of `LayerChoice` and `InputChoice` is shown below:
```json
{
    "key_name": {
        "_type": "layer_choice",
        "_value": ["op1_repr", "op2_repr", "op3_repr"]
    },
    "key_name": {
        "_type": "input_choice",
        "_value": {
            "candidates": ["in1_key", "in2_key", "in3_key"],
            "n_chosen": 1
        }
    }
}
```

Correspondingly, the generate architecture is in the following format:
```json
{
    "key_name": {
        "_value": "op1_repr",
        "_idx": 0
    },
    "key_name": {
        "_value": ["in2_key"],
        "_idex": [1]
    }
}
```
