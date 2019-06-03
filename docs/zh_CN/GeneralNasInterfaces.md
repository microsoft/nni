# 神经网络架构搜索的通用编程接口

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。 然而，实现这些算法需要很大的工作量，且很难重用其它算法的代码库来实现。

要促进 NAS 创新（例如，设计实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

## 编程接口

在两种场景下需要用于设计和搜索模型的新的编程接口。 1) 在设计神经网络时，层、子模型或连接有多个可能，并且不确定哪一个或哪种组合表现最好。 如果有一种简单的方法来表达想要尝试的候选层、子模型，将会很有价值。 2) 研究自动化 NAS 时，需要统一的方式来表达神经网络架构的搜索空间， 并在不改变 Trial 代码的情况下来使用不同的搜索算法。

本文基于 [NNI Annotation](./AnnotationSpec.md) 实现了简单灵活的编程接口 。 通过以下示例来详细说明。

### 示例：为层选择运算符

在设计此模型时，第四层的运算符有多个可能的选择，会让模型有更好的表现。 如图所示，在模型代码中可以对第四层使用 Annotation。 此 Annotation 中，共有五个字段：

![](../img/example_layerchoice.png)

* **layer_choice**：它是函数调用的 list，每个函数都要在代码或导入的库中实现。 函数的输入参数格式为：`def XXX (input, arg2, arg3, ...)`，其中输入是包含了两个元素的 list。 其中一个是 `fixed_inputs` 的 list，另一个是 `optional_inputs` 中选择输入的 list。 `conv` 和 `pool` 是函数示例。 对于 list 中的函数调用，无需写出第一个参数（即 input）。 注意，只会从这些函数调用中选择一个来执行。
* **fixed_inputs** ：它是变量的 list，可以是前一层输出的张量。 也可以是此层之前的另一个 `nni.mutable_layer` 的 `layer_output`，或此层之前的其它 Python 变量。 list 中的所有变量将被输入 `layer_choice` 中选择的函数（作为输入 list 的第一个元素）。
* **optional_inputs** ：它是变量的 list，可以是前一层的输出张量。 也可以是此层之前的另一个 `nni.mutable_layer` 的 `layer_output`，或此层之前的其它 Python 变量。 只有 `optional_input_size` 变量被输入 `layer_choice` 到所选的函数 （作为输入 list 的第二个元素）。
* **optional_input_size** ：它表示从 `input_candidates` 中选择多少个输入。 它可以是一个数字，也可以是一个范围。 范围 [1, 3] 表示选择 1、2 或 3 个输入。
* **layer_output** ：表示输出的名称。本例中，表示 `layer_choice` 选择的函数的返回值。 这是一个变量名，可以在随后的 Python 代码或 `nni.mutable_layer` 中使用。

此示例有两种写 Annotation 的方法。 对于上面的示例，输入函数的形式是 `[[], [out3]]` 。 对于下面的示例，输入的形式是 `[[out3], []]`。

### 示例：为层选择输入的连接

设计层的连接对于制作高性能模型至关重要。 通过此接口，可选择一个层可以采用哪些连接来作为输入。 可以从一组连接中选择几个。 下面的示例从三个候选输入中为 `concat` 这个函数选择两个输入 。 `concat` 还会使用 `fixed_inputs` 获取其上一层的输出 。

![](../img/example_connectchoice.png)

### 示例：同时选择运算符和连接

此示例从三个运算符中选择一个，并为其选择两个连接作为输入。 由于输入会有多个变量,，在函数的开头需要调用 `concat` 。

![](../img/example_combined.png)

### 示例：[ENAS](https://arxiv.org/abs/1802.03268) 宏搜索空间

为了证明编程接口带来的便利，使用该接口来实现 “ENAS + 宏搜索空间” 的 Trial 代码。 左图是 ENAS 论文中的宏搜索空间。

![](../img/example_enas.png)

## 统一的 NAS 搜索空间说明

通过上面的 Annotation 更新 Trial 代码后，即在代码中隐式指定了神经网络架构的搜索空间。 基于该代码，NNI 将自动生成一个搜索空间文件，可作为调优算法的输入。 搜索空间文件遵循以下 JSON 格式。

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

相应生成的神经网络结构（由调优算法生成）如下：

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

通过对搜索空间格式和体系结构选择 (choice) 表达式的说明，可以自由地在 NNI 上实现神经体系结构搜索的各种或通用的调优算法。 接下来的工作会提供一个通用的 NAS 算法。

=============================================================

## 神经网络结构搜索在 NNI 上的应用

### Experiment 执行的基本流程

NNI 的 Annotation 编译器会将 Trial 代码转换为可以接收架构选择并构建相应模型（如图）的代码。 NAS 的搜索空间可以看作是一个完整的图（在这里，完整的图意味着允许所有提供的操作符和连接来构建图），调优算法所选择的是其子图。 By default, the compiled trial code only builds and executes the subgraph.

![](../img/nas_on_nni.png)

The above figure shows how the trial code runs on NNI. `nnictl` processes user trial code to generate a search space file and compiled trial code. The former is fed to tuner, and the latter is used to run trials.

[**TODO**] Simple example of NAS on NNI.

### Weight sharing

Sharing weights among chosen architectures (i.e., trials) could speedup model search. For example, properly inheriting weights of completed trials could speedup the converge of new trials. One-Shot NAS (e.g., ENAS, Darts) is more aggressive, the training of different architectures (i.e., subgraphs) shares the same copy of the weights in full graph.

![](../img/nas_weight_share.png)

We believe weight sharing (transferring) plays a key role on speeding up NAS, while finding efficient ways of sharing weights is still a hot research topic. We provide a key-value store for users to store and load weights. Tuners and Trials use a provided KV client lib to access the storage.

[**TODO**] Example of weight sharing on NNI.

### Support of One-Shot NAS

One-Shot NAS is a popular approach to find good neural architecture within a limited time and resource budget. Basically, it builds a full graph based on the search space, and uses gradient descent to at last find the best subgraph. There are different training approaches, such as [training subgraphs (per mini-batch)](https://arxiv.org/abs/1802.03268), [training full graph through dropout](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf), [training with architecture weights (regularization)](https://arxiv.org/abs/1806.09055). Here we focus on the first approach, i.e., training subgraphs (ENAS).

With the same annotated trial code, users could choose One-Shot NAS as execution mode on NNI. Specifically, the compiled trial code builds the full graph (rather than subgraph demonstrated above), it receives a chosen architecture and training this architecture on the full graph for a mini-batch, then request another chosen architecture. It is supported by [NNI multi-phase](./multiPhase.md). We support this training approach because training a subgraph is very fast, building the graph every time training a subgraph induces too much overhead.

![](../img/one-shot_training.png)

The design of One-Shot NAS on NNI is shown in the above figure. One-Shot NAS usually only has one trial job with full graph. NNI supports running multiple such trial jobs each of which runs independently. As One-Shot NAS is not stable, running multiple instances helps find better model. Moreover, trial jobs are also able to synchronize weights during running (i.e., there is only one copy of weights, like asynchronous parameter-server mode). This may speedup converge.

[**TODO**] Example of One-Shot NAS on NNI.

## 通用的 NAS 调优算法

Like hyperparameter tuning, a relatively general algorithm for NAS is required. The general programming interface makes this task easier to some extent. We have a RL-based tuner algorithm for NAS from our contributors. We expect efforts from community to design and implement better NAS algorithms.

[**TODO**] More tuning algorithms for NAS.

## 导出最好的神经网络网络架构和代码

[**TODO**] After the NNI experiment is done, users could run `nnictl experiment export --code` to export the trial code with the best neural architecture.

## 结论和未来的工作

There could be different NAS algorithms and execution modes, but they could be supported with the same programming interface as demonstrated above.

There are many interesting research topics in this area, both system and machine learning.