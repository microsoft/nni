# 神经网络架构搜索的通用编程接口

自动神经架构搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动NAS的可行性，并且还发现了一些可以击败手动设计和调整模型的模型。 一些代表作品是[ NASNet ](https://arxiv.org/abs/1707.07012) ，[ ENAS ](https://arxiv.org/abs/1802.03268) ，[ DARTS ](https://arxiv.org/abs/1806.09055) ，[Network Morphism](https://arxiv.org/abs/1806.10282) ，和[Evolution](https://arxiv.org/abs/1703.01041)等 。 新的创新不断涌现。 然而，实现这些算法需要付出很大的努力，并且很难重用一种算法的代码库来实现另一种算法。

为了促进NAS创新（例如，设计/实施新的NAS模型，并排比较不同的NAS模型），易于使用且灵活的编程接口至关重要。

## 编程接口

在两种情况下经常需要使用用于设计和搜索模型的新编程接口。 1）在设计神经网络时，设计者可能对层，子模型或连接有多种选择，并且不确定哪一个或组合表现最佳。 如果能有一种简单的方法来表达他们想要尝试的候选层/子模型，将是很有吸引力的。 2）对于正在研究自动NAS的研究人员，他们希望有一种统一的方式来表达神经架构的搜索空间。 并使不变的试验代码适应不同的搜索算法。

我们基于[ NNI注释](./AnnotationSpec.md)设计了一个简单灵活的编程接口 。 通过以下示例详细说明。

###示例：为图层选择运算符

在设计以下模型时，第四层中可能有多个运算符选项可能使该模型表现良好。 在此模型的脚本中, 我们可以对第四层使用注释, 如图所示。 在此注释中，总共有五个字段：

![](../img/example_layerchoice.png)

* ** layer_choice ** ：它是一个函数调用列表，每个函数都应该在用户脚本或导入的库中定义。 函数的输入参数应遵循以下格式：` def XXX（输入，arg2，arg3，...） ` ，其中`输入`是一个包含两个元素的列表。 一个是` fixed_inputs的列表` ，另一个是来自` optional_inputs的所选输入的列表` 。 图中的` conv `和`pool`函数定义的示例。 对于此列表中的函数调用，无需编写第一个参数（即`input` ）。 请注意, 仅一个函数调用为此图层选择。
* ** fixed_inputs ** ：它是变量列表，变量可以是前一层的输出张量。 变量可以是此层之前的另一个nni.mutable_layer的` layer_output `，或此层之前的其他python变量。 此列表中的所有变量将被输入` layer_choice`中的所选函数 （作为`input`列表的第一个元素列表）。
* ** optional_inputs ** ：它是变量列表，变量可以是前一层的输出张量。 变量可以是此层之前的另一个nni.mutable_layer的` layer_output `，或此层之前的其他python变量。 只有` input_num `个变量将被输入` layer_choice`中的所选函数 （作为`input`列表的第二个元素列表）。
* ** optional_input_size ** ：它表示从` input_candidates` 中选择了多少输入。 它可以是一个数字, 也可以是一个范围。 范围 [1, 3] 表示它选择1、2或3个输入。
* ** layer_output ** ：此层的输出的名称，在这种情况下，它表示在` layer_choice`中函数调用 的返回值。 这将是一个变量名，可以在以下python代码或nni.mutable_layer中使用。

为此示例编写注释有两种方法。 对于上一个，输入`input `函数调用是` [[]，[out3]] ` 。 对于底部，输入` input`是` [[out3]，[]] ` 。

### 示例：选择图层的输入连接

设计层的连接对于制作高性能模型至关重要。 通过我们提供的接口，用户可以注释一个层采用哪些连接（作为输入）。 他们可以从一组连接中选择几个。 下面是一个例子，它从三个候选输入中为` concat`这个函数调用选择两个输入 。 这里` concat `始终使用` fixed_inputs`获取其上一层的输出 。

![](../img/example_connectchoice.png)

### 示例: 同时选择运算符和连接

在这个例子中，我们从三个运算符中选择一个，并为它选择两个连接。 由于`input`中有多个变量, 我们在函数的开头调用`concat` 。

![](../img/example_combined.png)

### 示例：[ ENAS ](https://arxiv.org/abs/1802.03268)宏搜索空间

为了说明编程接口的便利性，我们使用该接口来实现“ENAS +宏搜索空间”的试验代码。 左图是ENAS论文中的宏搜索空间。

![](../img/example_enas.png)

## 统一NAS搜索空间规范

通过上面的注释完成试验代码后，用户在代码中隐式指定了神经架构的搜索空间。 基于该代码, NNI 将自动生成一个搜索空间文件, 该文件可以输入到调优算法中。 此搜索空间文件遵循以下` json `格式。

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

NNI的注释编译器将带注释的试验代码转换为可以接收架构选择并构建相应模型（即图形）的代码。 The NAS search space can be seen as a full graph (here, full graph means enabling all the provided operators and connections to build a graph), the architecture chosen by the tuning algorithm is a subgraph in it. By default, the compiled trial code only builds and executes the subgraph.

![](../img/nas_on_nni.png)

The above figure shows how the trial code runs on NNI. `nnictl` processes user trial code to generate a search space file and compiled trial code. The former is fed to tuner, and the latter is used to run trilas.

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

The design of One-Shot NAS on NNI is shown in the above figure. One-Shot NAS usually only has one trial job with full graph. NNI supports running multiple such trial jobs each of which runs independently. As One-Shot NAS is not stable, running multiple instances helps find better model. Moreover, trial jobs are also able to synchronize weights during running (i.e., there is only one copy of weights, like asynchroneous parameter-server mode). This may speedup converge.

[**TODO**] Example of One-Shot NAS on NNI.

## General tuning algorithms for NAS

Like hyperparameter tuning, a relatively general algorithm for NAS is required. The general programming interface makes this task easier to some extent. We have a RL-based tuner algorithm for NAS from our contributors. We expect efforts from community to design and implement better NAS algorithms.

[**TODO**] More tuning algorithms for NAS.

## Export best neural architecture and code

[**TODO**] After the NNI experiment is done, users could run `nnictl experiment export --code` to export the trial code with the best neural architecture.

## Conclusion and Future work

There could be different NAS algorithms and execution modes, but they could be supported with the same programming interface as demonstrated above.

There are many interesting research topics in this area, both system and machine learning.