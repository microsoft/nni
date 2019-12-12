# NNI NAS 编程接口

我们正在尝试通过统一的编程接口来支持各种 NAS 算法，当前处于试验阶段。 这意味着当前编程接口可能会进行重大变化。

## 模型的编程接口

The programming interface of designing and searching a model is often demanded in two scenarios.

1. 在设计神经网络时，可能在层、子模型或连接上有多种选择，并且无法确定是其中一种或某些的组合的结果最好。 因此，需要简单的方法来表达候选的层或子模型。
2. 在神经网络上应用 NAS 时，需要统一的方式来表达架构的搜索空间，这样不必为不同的搜索算法来更改代码。


For expressing neural architecture search space in user code, we provide the following APIs (take PyTorch as example):

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
This is for users to specify multiple candidate operations for a layer, one operation will be chosen at last. `key` is the identifier of the layer,it could be used to share choice between multiple `LayerChoice`. For example, there are two `LayerChoice` with the same candidate operations, and you want them to have the same choice (i.e., if first one chooses the `i`th op, the second one also chooses the `i`th op), give them the same key.

```python
def __init__(self):
    ...
    # 从 ``n_candidates`` 个输入中选择 ``n_selected`` 个。
    # n_candidates: 候选输入数量
    # n_chosen: 选择的数量
    # key: ``InputChoice`` 实例的名称
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


## 两种训练模式

After writing your model with search space embedded in the model using the above APIs, the next step is finding the best model from the search space. There are two training modes: [one-shot training mode](#one-shot-training-mode) and [classic distributed search](#classic-distributed-search).

### One-shot 训练模式

Similar to optimizers of deep learning models, the procedure of finding the best model from search space can be viewed as a type of optimizing process, we call it `NAS trainer`. There have been several NAS trainers, for example, `DartsTrainer` which uses SGD to train architecture weights and model weights iteratively, `ENASTrainer` which uses a controller to train the model. New and more efficient NAS trainers keep emerging in research community.

NNI provides some popular NAS trainers, to use a NAS trainer, users could initialize a trainer after the model is defined:

```python
# 创建 DartsTrainer
trainer = DartsTrainer(model,
                       loss=criterion,
                       metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                       optimizer=optim,
                       num_epochs=args.epochs,
                       dataset_train=dataset_train,
                       dataset_valid=dataset_valid,)
# 从搜索空间中找到最好的模型
trainer.train()
# 导出最好的模型
trainer.export(file='./chosen_arch')
```

Different trainers could have different input arguments depending on their algorithms. Please refer to [each trainer's code](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch) for detailed arguments. After training, users could export the best one of the found models through `trainer.export()`. No need to start an NNI experiment through `nnictl`.

The supported trainers can be found [here](Overview.md#supported-one-shot-nas-algorithms). A very simple example using NNI NAS API can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/simple/train.py).

### 经典分布式搜索

Neural architecture search is originally executed by running each child model independently as a trial job. We also support this searching approach, and it naturally fits in NNI hyper-parameter tuning framework, where tuner generates child model for next trial and trials run in training service.

For using this mode, no need to change the search space expressed with NNI NAS API (i.e., `LayerChoice`, `InputChoice`, `MutableScope`). After the model is initialized, apply the function `get_and_apply_next_architecture` on the model. One-shot NAS trainers are not used in this mode. Here is a simple example:
```python
class Net(nn.Module):
    # 使用 LayerChoice 和 InputChoice 的模型
    ...

model = Net()
# 从 Tuner 中选择架构，并应用到模型上
get_and_apply_next_architecture(model)
# 训练模型
train(model)
# 测试模型
acc = test(model)
# 返回此架构的性能
nni.report_final_result(acc)
```

The search space should be automatically generated and sent to tuner. As with NNI NAS API the search space is embedded in user code, users could use "[nnictl ss_gen](../Tutorial/Nnictl.md)" to generate search space file. Then, put the path of the generated search space in the field `searchSpacePath` of `config.yml`. The other fields in `config.yml` can be filled by referring [this tutorial](../Tutorial/QuickStart.md).

You could use [NNI tuners](../Tuner/BuiltinTuner.md) to do the search.

We support standalone mode for easy debugging, where you could directly run the trial command without launching an NNI experiment. This is for checking whether your trial code can correctly run. The first candidate(s) are chosen for `LayerChoice` and `InputChoice` in this standalone mode.

The complete example code can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/classic_nas/config_nas.yml).

## NAS 算法的编程接口

We also provide simple interface for users to easily implement a new NAS trainer on NNI.

### 在 NNI 上实现新的 NAS Trainer

To implement a new NAS trainer, users basically only need to implement two classes by inheriting `BaseMutator` and `BaseTrainer` respectively.

In `BaseMutator`, users need to overwrite `on_forward_layer_choice` and `on_forward_input_choice`, which are the implementation of `LayerChoice` and `InputChoice` respectively. Users could use property `mutables` to get all `LayerChoice` and `InputChoice` in the model code. Then users need to implement a new trainer, which instantiates the new mutator and implement the training logic. For details, please read [the code](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch) and the supported trainers, for example, [DartsTrainer](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch/darts).

### 为 NAS 实现 NNI Tuner

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
