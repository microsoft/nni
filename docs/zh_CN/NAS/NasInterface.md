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
用户可为某层指定多个候选的操作，最后从其中选择一个。 `key` 是层的标识符，可被用来在多个 `LayerChoice` 间共享选项。 例如，两个 `LayerChoice` 有相同的候选操作，并希望能使用同样的选择，(即，如果第一个选择了第 `i` 个操作，第二个也应该选择第 `i` 个操作)，则可给它们相同的 key。

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
`InputChoice` 是一个 PyTorch module，初始化时需要元信息，例如，从多少个输入后选中选择多少个输入，初始化的 `InputChoice` 名称。 真正候选的输入张量只能在 `forward` 函数中获得。 在 `InputChoice` 中，`forward` 会在调用时传入实际的候选输入张量。

一些 [NAS Trainer](#one-shot-training-mode) 需要知道输入张量的来源层，因此在 `InputChoice` 中添加了输入参数 `choose_from` 来表示每个候选输入张量的来源层。 `choose_from` 是 str 的 list，每个元素都是 `LayerChoice` 和`InputChoice` 的 `key`，或者 module 的 name (详情参考[代码](https://github.com/microsoft/nni/blob/master/src/sdk/pynni/nni/nas/pytorch/mutables.py))。


除了 `LayerChoice` 和 `InputChoice`，还提供了 `MutableScope`，可以让用户标记自网络，从而给 NAS Trainer 提供更多的语义信息 (如网络结构)。 示例如下：
```python
class Cell(mutables.MutableScope):
    def __init__(self, scope_name):
        super().__init__(scope_name)
        self.layer1 = nni.nas.pytorch.LayerChoice(...)
        self.layer2 = nni.nas.pytorch.LayerChoice(...)
        self.layer3 = nni.nas.pytorch.LayerChoice(...)
        ...
```
名为 `scope_name` 的 `MutableScope` 包含了三个 `LayerChoice` 层 (`layer1`, `layer2`, `layer3`)。 NAS Trainer 可获得这样的分层结构。


## 两种训练模式

在使用上述 API 在模型中嵌入 搜索空间后，下一步是从搜索空间中找到最好的模型。 有两种驯良模式：[one-shot 训练模式](#one-shot-training-mode) and [经典的分布式搜索](#classic-distributed-search)。

### One-shot 训练模式

与深度学习模型的优化器相似，从搜索空间中找到最好模型的过程可看作是优化过程，称之为 `NAS Trainer`。 NAS Trainer 包括 `DartsTrainer` 使用了 SGD 来交替训练架构和模型权重，`ENASTrainer` 使用 Controller 来训练模型。 新的、更高效的 NAS Trainer 在研究界不断的涌现出来。

NNI 提供了一些流行的 NAS Trainer，要使用 NAS Trainer，用户需要在模型定义后初始化 Trainer：

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

不同的 Trainer 可能有不同的输入参数，具体取决于其算法。 详细参数可参考具体的 [Trainer 代码](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/nas/pytorch)。 训练完成后，可通过 `trainer.export()` 导出找到的最好的模型。 No need to start an NNI experiment through `nnictl`.

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

### 为 NAS 实现 NNI Tuner

NNI 中的 NAS Tuner 需要自动生成搜索空间。 `LayerChoice` 和 `InputChoice` 的搜索空间格式如下：
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

相应的，生成的网络架构格式如下：
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