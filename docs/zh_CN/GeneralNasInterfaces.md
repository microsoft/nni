# 神经网络架构搜索的通用编程接口

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。 然而，实现这些算法需要很大的工作量，且很难重用其它算法的代码库来实现。

要促进 NAS 创新（例如，设计实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

## 编程接口

在两种场景下需要用于设计和搜索模型的新的编程接口。 1) 在设计神经网络时，层、子模型或连接有多个可能，并且不确定哪一个或哪种组合表现最好。 如果有一种简单的方法来表达想要尝试的候选层、子模型，将会很有价值。 2) 研究自动化 NAS 时，需要统一的方式来表达神经网络架构的搜索空间， 并在不改变 Trial 代码的情况下来使用不同的搜索算法。

本文基于 [NNI Annotation](./AnnotationSpec.md) 实现了简单灵活的编程接口 。 通过以下示例来详细说明。

### 示例：为层选择运算符

在设计此模型时，第四层的运算符有多个可能的选择，会让模型有更好的表现。 如图所示，在模型代码中可以对第四层使用 Annotation。 此 Annotation 中，共有五个字段：

![](../img/example_layerchoice.png)

* **layer_choice**：它是函数调用的 list，每个函数都要在代码或导入的库中实现。 函数的输入参数格式为：`def XXX (input, arg2, arg3, ...)`，其中 `input` 是包含了两个元素的 list。 其中一个是 `fixed_inputs` 的 list，另一个是 `optional_inputs` 中选择输入的 list。 `conv` 和 `pool` 是函数示例。 对于 list 中的函数调用，无需写出第一个参数（即 `input`）。 注意，只会从这些函数调用中选择一个来执行。
* **fixed_inputs** ：它是变量的 list，可以是前一层输出的张量。 也可以是此层之前的另一个 nni.mutable_layer 的 `layer_output`，或此层之前的其它 Python 变量。 list 中的所有变量将被输入 `layer_choice` 中选择的函数（作为 `input` list 的第一个元素）。
* **optional_inputs** ：它是变量的 list，可以是前一层的输出张量。 也可以是此层之前的另一个 nni.mutable_layer 的 `layer_output`，或此层之前的其它 Python 变量。 只有 `input_num` 变量被输入 `layer_choice` 到所选的函数 （作为 `input` list 的第二个元素）。
* **optional_input_size** ：它表示从 `input_candidates` 中选择多少个输入。 它可以是一个数字，也可以是一个范围。 范围 [1, 3] 表示选择 1、2 或 3 个输入。
* **layer_output** ：表示输出的名称。本例中，表示 `layer_choice` 选择的函数的返回值。 这是一个变量名，可以在随后的 Python 代码或 nni.mutable_layer 中使用。

为此示例编写注释有两种方法。 对于上一个，输入 `input` 函数调用是`[[]，[out3]] ` 。 对于底部，输入 `input` 是`[[out3]，[]]` 。

### 示例：选择图层的输入连接

设计层的连接对于制作高性能模型至关重要。 通过我们提供的接口，用户可以注释一个层采用哪些连接（作为输入）。 他们可以从一组连接中选择几个。 下面是一个例子，它从三个候选输入中为 `concat` 这个函数调用选择两个输入 。 这里 `concat` 始终使用 `fixed_inputs` 获取其上一层的输出 。

![](../img/example_connectchoice.png)

### 示例: 同时选择运算符和连接

在这个例子中，我们从三个运算符中选择一个，并为它选择两个连接。 由于 `input` 中有多个变量, 我们在函数的开头调用 `concat` 。

![](../img/example_combined.png)

### 示例：[ENAS](https://arxiv.org/abs/1802.03268) 宏搜索空间

为了说明编程接口的便利性，我们使用该接口来实现“ENAS +宏搜索空间”的试验代码。 左图是 ENAS 论文中的宏搜索空间。

![](../img/example_enas.png)

## 统一 NAS 搜索空间规范

通过上面的注释完成试验代码后，用户在代码中隐式指定了神经架构的搜索空间。 基于该代码，NNI 将自动生成一个搜索空间文件, 该文件可以输入到调优算法中。 此搜索空间文件遵循以下` json `格式。

```json
{
    "mutable_1": {
        "layer_1": {
            "layer_choice": ["conv(ch=128)", "pool", "identity"],
            "optional_inputs": ["out1", "out2", "out3"],
            "optional_input_size": 2
        },
        "layer_2": {
            ...
        }
    }
}
```

因此, 指定的神经体系结构 (由调优算法生成) 表示如下:

```json
{
    "mutable_1": {
        "layer_1": {
            "chosen_layer": "pool",
            "chosen_inputs": ["out1", "out3"]
        },
        "layer_2": {
            ...
        }
    }
}
```

通过对搜索空间格式和体系结构 (选择) 表达式的规范, 用户可以自由地在 NNI 上实现神经体系结构搜索的各种 (一般) 调优算法。 今后的一项工作是提供一个通用 NAS 算法。

=============================================================

## 神经结构搜索在 NNI 上的应用

### 实验执行的基本流程

NNI 的注释编译器将带注释的试验代码转换为可以接收架构选择并构建相应模型（即图）的代码。 NAS 搜索空间可以看作是一个完整的图形（这里，完整的图形意味着允许所有提供的操作符和连接来构建图形），调整算法选择的体系结构是其中的子图形。 默认情况下，编译的试验代码仅构建并执行子图。

![](../img/nas_on_nni.png)

上图显示了试验代码如何在NNI上运行。 `nnictl` 处理用户试用代码以生成搜索空间文件和编译的试用代码。 前者用于 Tuner，后者用于运行 Trial。

[**TODO**] NNI 上 NAS 的简单示例。

### 权重共享

在所选择的架构（即试验）之间共享权重可以加速模型搜索。 例如，适当地继承已完成试验的权重可以加速新试验的收敛。 One-shot NAS（例如，ENAS，Darts）更具侵略性，不同架构（即子图）的训练在完整图中共享相同的权重副本。

![](../img/nas_weight_share.png)

我们认为权重分配（转移）在加速 NAS 方面起着关键作用，而找到有效的权重共享方式仍然是一个热门的研究课题。 我们为用户提供了一个键值存储, 用于存储和加载权重。 Tuner 和 Trial 使用提供的 KV 客户端库来访问存储。

[**TODO**] NNI上的权重分享示例。

### 支持One-Shot NAS

One-Shot NAS是一种在有限的时间和资源预算内找到良好的神经结构的流行方法。 基本上，它基于搜索空间构建完整的图形，并使用梯度下降最终找到最佳子图。 有不同的训练方法，例如：[training subgraphs (per mini-batch)](https://arxiv.org/abs/1802.03268) ，[training full graph through dropout](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf)，以及 [training with architecture weights (regularization)](https://arxiv.org/abs/1806.09055) 。 在这里，我们关注第一种方法，即训练子图（ENAS）。

使用相同的带注释的试验代码，用户可以选择 One-Shot NAS 作为 NNI 上的执行模式。 具体来说, 编译后的试用代码构建完整的图形 (而不是上面演示的子图), 它接收所选择的架构, 并在完整的图形上对此体系结构进行小型批处理的训练, 然后请求另一个选定的架构。 由 [NNI 多阶段](./multiPhase.md) 支持。 因为训练子图非常快，而每次训练子图时都会产生过多的开销，所以支持这种训练方法。

![](../img/one-shot_training.png)

One-Shot NAS 的设计如上图所示。 One-Shot NAS 通常只有一个带有完整图形的试验作业。 NNI支持运行多个此类试验作业，每个作业都独立运行。 由于 One-Shot NAS 不稳定，运行多个实例有助于找到更好的模型。 此外，试运行也能够在运行期间同步权重（即，只有一个权重副本，如异步参数 - 服务器模式）。 这可能会加速收敛。

[**TODO**] NNI 上的权重分享示例。

## 通用的 NAS 调优算法

与超参数调整一样, NAS 也需要相对通用的算法。 通用编程接口使这项任务在某种程度上更容易。 我们的贡献者为 NAS 提供了基于 RL 的调参算法。 我们期待社区努力设计和实施更好的 NAS 调优算法。

[**TODO**] 更多 NAS 的调优算法。

## 导出最好的神经网络架构和代码

[**TODO**] Experiment 完成后，可以运行 `nnictl experiment export --code` 来导出用最好的神经结构导出试验代码。

## 结论和未来的工作

可能有不同的 NAS 算法和执行模式, 但它们可以通过相同的编程接口得到支持, 如上面所示。

在这一领域有许多有趣的研究主题, 包括系统和机器学习。