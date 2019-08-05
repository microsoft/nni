# 神经网络架构搜索的通用编程接口（测试版）

** 这是一个测试中的功能，目前只实现了通用的 NAS 编程接口。 Weight sharing will be supported in the following releases.*

自动化的神经网络架构（NAS）搜索在寻找更好的模型方面发挥着越来越重要的作用。 最近的研究工作证明了自动化 NAS 的可行性，并发现了一些超越手动设计和调整的模型。 代表算法有 [NASNet](https://arxiv.org/abs/1707.07012)，[ENAS](https://arxiv.org/abs/1802.03268)，[DARTS](https://arxiv.org/abs/1806.09055)，[Network Morphism](https://arxiv.org/abs/1806.10282)，以及 [Evolution](https://arxiv.org/abs/1703.01041) 等。 新的算法还在不断涌现。 然而，实现这些算法需要很大的工作量，且很难重用其它算法的代码库来实现。

要促进 NAS 创新（例如，设计实现新的 NAS 模型，并列比较不同的 NAS 模型），易于使用且灵活的编程接口非常重要。

<a name="ProgInterface"></a>

## 编程接口

A new programming interface for designing and searching for a model is often demanded in two scenarios. 1) When designing a neural network, the designer may have multiple choices for a layer, sub-model, or connection, and not sure which one or a combination performs the best. It would be appealing to have an easy way to express the candidate layers/sub-models they want to try. 2) For the researchers who are working on automatic NAS, they want to have an unified way to express the search space of neural architectures. And making unchanged trial code adapted to different searching algorithms.

We designed a simple and flexible programming interface based on [NNI annotation](../Tutorial/AnnotationSpec.md). It is elaborated through examples below.

### 示例：为层选择运算符

When designing the following model there might be several choices in the fourth layer that may make this model perform well. In the script of this model, we can use annotation for the fourth layer as shown in the figure. In this annotation, there are five fields in total:

![](../../img/example_layerchoice.png)

* **layer_choice**：它是函数调用的 list，每个函数都要在代码或导入的库中实现。 函数的输入参数格式为：`def XXX (input, arg2, arg3, ...)`，其中输入是包含了两个元素的 list。 其中一个是 `fixed_inputs` 的 list，另一个是 `optional_inputs` 中选择输入的 list。 `conv` 和 `pool` 是函数示例。 对于 list 中的函数调用，无需写出第一个参数（即 input）。 注意，只会从这些函数调用中选择一个来执行。
* **fixed_inputs** ：它是变量的 list，可以是前一层输出的张量。 也可以是此层之前的另一个 `nni.mutable_layer` 的 `layer_output`，或此层之前的其它 Python 变量。 list 中的所有变量将被输入 `layer_choice` 中选择的函数（作为输入 list 的第一个元素）。
* **optional_inputs** ：它是变量的 list，可以是前一层的输出张量。 也可以是此层之前的另一个 `nni.mutable_layer` 的 `layer_output`，或此层之前的其它 Python 变量。 只有 `optional_input_size` 变量被输入 `layer_choice` 到所选的函数 （作为输入 list 的第二个元素）。
* **optional_input_size** ：它表示从 `input_candidates` 中选择多少个输入。 它可以是一个数字，也可以是一个范围。 范围 [1, 3] 表示选择 1、2 或 3 个输入。
* **layer_output** ：表示输出的名称。本例中，表示 `layer_choice` 选择的函数的返回值。 这是一个变量名，可以在随后的 Python 代码或 `nni.mutable_layer` 中使用。

There are two ways to write annotation for this example. For the upper one, input of the function calls is `[[],[out3]]`. For the bottom one, input is `[[out3],[]]`.

**Debugging**: We provided an `nnictl trial codegen` command to help debugging your code of NAS programming on NNI. If your trial with trial_id `XXX` in your experiment `YYY` is failed, you could run `nnictl trial codegen YYY --trial_id XXX` to generate an executable code for this trial under your current directory. With this code, you can directly run the trial command without NNI to check why this trial is failed. Basically, this command is to compile your trial code and replace the NNI NAS code with the real chosen layers and inputs.

### 示例：为层选择输入的连接

Designing connections of layers is critical for making a high performance model. With our provided interface, users could annotate which connections a layer takes (as inputs). They could choose several ones from a set of connections. Below is an example which chooses two inputs from three candidate inputs for `concat`. Here `concat` always takes the output of its previous layer using `fixed_inputs`.

![](../../img/example_connectchoice.png)

### 示例：同时选择运算符和连接

In this example, we choose one from the three operators and choose two connections for it. As there are multiple variables in inputs, we call `concat` at the beginning of the functions.

![](../../img/example_combined.png)

### 示例：[ENAS](https://arxiv.org/abs/1802.03268) 宏搜索空间

To illustrate the convenience of the programming interface, we use the interface to implement the trial code of "ENAS + macro search space". The left figure is the macro search space in ENAS paper.

![](../../img/example_enas.png)

## 统一的 NAS 搜索空间说明

After finishing the trial code through the annotation above, users have implicitly specified the search space of neural architectures in the code. Based on the code, NNI will automatically generate a search space file which could be fed into tuning algorithms. This search space file follows the following JSON format.

```javascript
{
    "mutable_1": {
        "_type": "mutable_layer",
        "_value": {
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
}
```

Accordingly, a specified neural architecture (generated by tuning algorithm) is expressed as follows:

```javascript
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

With the specification of the format of search space and architecture (choice) expression, users are free to implement various (general) tuning algorithms for neural architecture search on NNI. One future work is to provide a general NAS algorithm.

## Support of One-Shot NAS

One-Shot NAS is a popular approach to find good neural architecture within a limited time and resource budget. Basically, it builds a full graph based on the search space, and uses gradient descent to at last find the best subgraph. There are different training approaches, such as [training subgraphs (per mini-batch)](https://arxiv.org/abs/1802.03268), [training full graph through dropout](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf), [training with architecture weights (regularization)](https://arxiv.org/abs/1806.09055).

NNI has supported the general NAS as demonstrated above. From users' point of view, One-Shot NAS and NAS have the same search space specification, thus, they could share the same programming interface as demonstrated above, just different training modes. NNI provides four training modes:

***classic_mode***: this mode is described [above](#ProgInterface), in this mode, each subgraph runs as a trial job. To use this mode, you should enable NNI annotation and specify a tuner for nas in experiment config file. [Here](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas) is an example to show how to write trial code and the config file. And [here](https://github.com/microsoft/nni/tree/master/examples/tuners/random_nas_tuner) is a simple tuner for nas.

***enas_mode***: following the training approach in [enas paper](https://arxiv.org/abs/1802.03268). It builds the full graph based on neural architrecture search space, and only activate one subgraph that generated by the controller for each mini-batch. [Detailed Description](#ENASMode). (currently only supported on tensorflow).

To use enas_mode, you should add one more field in the `trial` config as shown below.

```diff
trial:
    command: your command to run the trial
    codeDir: the directory where the trial's code is located
    gpuNum: the number of GPUs that one trial job needs

+   #choice: classic_mode, enas_mode, oneshot_mode
+   nasMode: enas_mode
```

Similar to classic_mode, in enas_mode you need to specify a tuner for nas, as it also needs to receive subgraphs from tuner (or controller using the terminology in the paper). Since this trial job needs to receive multiple subgraphs from tuner, each one for a mini-batch, two lines need to be added to the trial code to receive the next subgraph (i.e., `nni.training_update`) and report the result of the current subgraph. Below is an example:

```python
for _ in range(num):
    # here receives and enables a new subgraph
    """@nni.training_update(tf=tf, session=self.session)"""
    loss, _ = self.session.run([loss_op, train_op])
    # report the loss of this mini-batch
    """@nni.report_final_result(loss)"""
```

Here, `nni.training_update` is to do some update on the full graph. In enas_mode, the update means receiving a subgraph and enabling it on the next mini-batch. While in darts_mode, the update means training the architecture weights (details in darts_mode). In enas_mode, you need to pass the imported tensorflow package to `tf` and the session to `session`.

***oneshot_mode***: following the training approach in [this paper](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf). Different from enas_mode which trains the full graph by training large numbers of subgraphs, in oneshot_mode the full graph is built and dropout is added to candidate inputs and also added to candidate ops' outputs. Then this full graph is trained like other DL models. [Detailed Description](#OneshotMode). (currently only supported on tensorflow).

To use oneshot_mode, you should add one more field in the `trial` config as shown below. In this mode, no need to specify tuner in the config file as it does not need tuner. (Note that you still need to specify a tuner (any tuner) in the config file for now.) Also, no need to add `nni.training_update` in this mode, because no special processing (or update) is needed during training.

```diff
trial:
    command: your command to run the trial
    codeDir: the directory where the trial's code is located
    gpuNum: the number of GPUs that one trial job needs

+   #choice: classic_mode, enas_mode, oneshot_mode
+   nasMode: oneshot_mode
```

***darts_mode***: following the training approach in [this paper](https://arxiv.org/abs/1806.09055). It is similar to oneshot_mode. There are two differences, one is that darts_mode only add architecture weights to the outputs of candidate ops, the other is that it trains model weights and architecture weights in an interleaved manner. [Detailed Description](#DartsMode).

To use darts_mode, you should add one more field in the `trial` config as shown below. In this mode, also no need to specify tuner in the config file as it does not need tuner. (Note that you still need to specify a tuner (any tuner) in the config file for now.)

```diff
trial:
    command: your command to run the trial
    codeDir: the directory where the trial's code is located
    gpuNum: the number of GPUs that one trial job needs

+   #choice: classic_mode, enas_mode, oneshot_mode
+   nasMode: darts_mode
```

When using darts_mode, you need to call `nni.training_update` as shown below when architecture weights should be updated. Updating architecture weights need `loss` for updating the weights as well as the training data (i.e., `feed_dict`) for it.

```python
for _ in range(num):
    # here trains the architecture weights
    """@nni.training_update(tf=tf, session=self.session, loss=loss, feed_dict=feed_dict)"""
    loss, _ = self.session.run([loss_op, train_op])
```

**Note:** for enas_mode, oneshot_mode, and darts_mode, NNI only works on the training phase. They also have their own inference phase which is not handled by NNI. For enas_mode, the inference phase is to generate new subgraphs through the controller. For oneshot_mode, the inference phase is sampling new subgraphs randomly and choosing good ones. For darts_mode, the inference phase is pruning a proportion of candidates ops based on architecture weights.

<a name="ENASMode"></a>

### enas_mode

In enas_mode, the compiled trial code builds the full graph (rather than subgraph), it receives a chosen architecture and training this architecture on the full graph for a mini-batch, then request another chosen architecture. It is supported by [NNI multi-phase](./multiPhase.md).

Specifically, for trials using tensorflow, we create and use tensorflow variable as signals, and tensorflow conditional functions to control the search space (full-graph) to be more flexible, which means it can be changed into different sub-graphs (multiple times) depending on these signals. [Here]() is an example for enas_mode.

<a name="OneshotMode"></a>

### oneshot_mode

Below is the figure to show where dropout is added to the full graph for one layer in `nni.mutable_layers`, input 1-k are candidate inputs, the four ops are candidate ops.

![](../../img/oneshot_mode.png)

As suggested in the [paper](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf), a dropout method is implemented to the inputs for every layer. The dropout rate is set to r^(1/k), where 0 < r < 1 is a hyper-parameter of the model (default to be 0.01) and k is number of optional inputs for a specific layer. The higher the fan-in, the more likely each possible input is to be dropped out. However, the probability of dropping out all optional_inputs of a layer is kept constant regardless of its fan-in. Suppose r = 0.05. If a layer has k = 2 optional_inputs then each one will independently be dropped out with probability 0.051/2 ≈ 0.22 and will be retained with probability 0.78. If a layer has k = 7 optional_inputs then each one will independently be dropped out with probability 0.051/7 ≈ 0.65 and will be retained with probability 0.35. In both cases, the probability of dropping out all of the layer's optional_inputs is 5%. The outputs of candidate ops are dropped out through the same way. [Here]() is an example for oneshot_mode.

<a name="DartsMode"></a>

### darts_mode

Below is the figure to show where architecture weights are added to the full graph for one layer in `nni.mutable_layers`, output of each candidate op is multiplied by a weight which is called architecture weight.

![](../../img/darts_mode.png)

In `nni.training_update`, tensorflow MomentumOptimizer is used to train the architecture weights based on the pass `loss` and `feed_dict`. [Here]() is an example for darts_mode.

### [**TODO**] Multiple trial jobs for One-Shot NAS

One-Shot NAS usually has only one trial job with the full graph. However, running multiple such trial jobs leads to benefits. For example, in enas_mode multiple trial jobs could share the weights of the full graph to speedup the model training (or converge). Some One-Shot approaches are not stable, running multiple trial jobs increase the possibility of finding better models.

NNI natively supports running multiple such trial jobs. The figure below shows how multiple trial jobs run on NNI.

![](../../img/one-shot_training.png)

=============================================================

## System design of NAS on NNI

### Basic flow of experiment execution

NNI's annotation compiler transforms the annotated trial code to the code that could receive architecture choice and build the corresponding model (i.e., graph). The NAS search space can be seen as a full graph (here, full graph means enabling all the provided operators and connections to build a graph), the architecture chosen by the tuning algorithm is a subgraph in it. By default, the compiled trial code only builds and executes the subgraph.

![](../../img/nas_on_nni.png)

The above figure shows how the trial code runs on NNI. `nnictl` processes user trial code to generate a search space file and compiled trial code. The former is fed to tuner, and the latter is used to run trials.

[Simple example of NAS on NNI](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas).

### [**TODO**] Weight sharing

Sharing weights among chosen architectures (i.e., trials) could speedup model search. For example, properly inheriting weights of completed trials could speedup the converge of new trials. One-Shot NAS (e.g., ENAS, Darts) is more aggressive, the training of different architectures (i.e., subgraphs) shares the same copy of the weights in full graph.

![](../../img/nas_weight_share.png)

We believe weight sharing (transferring) plays a key role on speeding up NAS, while finding efficient ways of sharing weights is still a hot research topic. We provide a key-value store for users to store and load weights. Tuners and Trials use a provided KV client lib to access the storage.

Example of weight sharing on NNI.

## General tuning algorithms for NAS

Like hyperparameter tuning, a relatively general algorithm for NAS is required. The general programming interface makes this task easier to some extent. We have an [RL tuner based on PPO algorithm](https://github.com/microsoft/nni/tree/master/src/sdk/pynni/nni/ppo_tuner) for NAS. We expect efforts from community to design and implement better NAS algorithms.

## [**TODO**] Export best neural architecture and code

After the NNI experiment is done, users could run `nnictl experiment export --code` to export the trial code with the best neural architecture.

## Conclusion and Future work

There could be different NAS algorithms and execution modes, but they could be supported with the same programming interface as demonstrated above.

There are many interesting research topics in this area, both system and machine learning.