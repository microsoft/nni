# NNI 中的 PPO Tuner

## PPOTuner

这是通常用于 NAS 接口的 NNI Tuner，使用了 [PPO 算法](https://arxiv.org/abs/1707.06347)。 此实现继承了[这里](https://github.com/openai/baselines/tree/master/baselines/ppo2)的主要逻辑，(即 OpenAI 的 PPO2)，并为 NAS 场景做了适配。

它能成功调优 [mnist-nas 示例](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas)，结果如下：

![](../../img/ppo_mnist.png)

We also tune [the macro search space for image classification in the enas paper](https://github.com/microsoft/nni/tree/master/examples/trials/nas_cifar10) (with limited epoch number for each trial, i.e., 8 epochs), which is implemented using the NAS interface and tuned with PPOTuner. Use Figure 7 in the [enas paper](https://arxiv.org/pdf/1802.03268.pdf) to show how the search space looks like

![](../../img/enas_search_space.png)

The figure above is a chosen architecture, we use it to show how the search space looks like. Each square is a layer whose operation can be chosen from 6 operations. Each dash line is a skip connection, each square layer could choose 0 or 1 skip connection getting the output of a previous layer. **Note that** in original macro search space each square layer could choose any number of skip connections, while in our implementation it is only allowed to choose 0 or 1.

The result is shown in figure below (with the experiment config [here](https://github.com/microsoft/nni/blob/master/examples/trials/nas_cifar10/config_ppo.yml)):

![](../../img/ppo_cifar10.png)