# NNI NAS 编程接口

我们正在尝试通过统一的编程接口来支持各种 NAS 算法，当前处于试验阶段。 这意味着当前编程接口可能会进行重大变化。

*先前的 [NAS annotation](../AdvancedFeature/GeneralNasInterfaces.md) 接口会很快被弃用。*

## 模型的编程接口

在两种场景下需要用于设计和搜索模型的编程接口。

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。


在用户代码中表示的神经网络搜索空间，可使用以下 API （以 PyTorch 为例）：

```python
# 在 PyTorch module 类中
def __init__(self):
    ...
    # 从 ``ops`` 中选择 ``ops``, 这是 PyTorch 中的 module。
    # op_candidates: 在 PyTorch 中 ``ops`` 是 module 的 list，而在 TensorFlow 中是 Keras 层的 list。
    # key: ``LayerChoice`` 实例的名称
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
用户可为某层指定多个候选的操作，最后从其中选择一个。 `key` 是层的标识符，可被用来在多个 `LayerChoice` 间共享选项。 For example, there are two `LayerChoice` with the same candidate operations, and you want them to have the same choice (i.e., if first one chooses the `i`th op, the second one also chooses the `i`th op), give them the same key.

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
`InputChoice` is a PyTorch module, in init, it needs meta information, for example, from how many input candidates to choose how many inputs, the name of this initialized `InputChoice`. The real candidate input tensors can only be obtained in `forward` function. In `forward`, `InputChoice` instance is called with real candidate input tensors.

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

The supported trainers can be found [here](./Overview.md#supported-one-shot-nas-algorithms). A very simple example using NNI NAS API can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/simple/train.py).

The complete example code can be found [here]().

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