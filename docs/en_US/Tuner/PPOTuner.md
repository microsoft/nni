PPO Tuner on NNI
===

## PPOTuner

This is a tuner generally for NNI's NAS interface, it uses [ppo algorithm](https://arxiv.org/abs/1707.06347). The implementation inherits the main logic of the implementation [here](https://github.com/openai/baselines/tree/master/baselines/ppo2) (i.e., ppo2 from OpenAI), and is adapted for NAS scenario.

It could successfully tune the [mnist-nas example](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas), and has the following result:

![](../../img/ppo_mnist.png)

We also tune [the macro search space for image classification in the enas paper](https://github.com/microsoft/nni/tree/master/examples/trials/nas_cifar10) (with limited epoch number for each trial, i.e., 8 epochs), which is implemented using the NAS interface and tuned with PPOTuner. Use Figure 7 in the [enas paper](https://arxiv.org/pdf/1802.03268.pdf) to show how the search space looks like

![](../../img/enas_search_space.png)

The figure above is a chosen architecture, we use it to show how the search space looks like. Each square is a layer whose operation can be chosen from 6 operations. Each dash line is a skip connection, each square layer could choose 0 or 1 skip connection getting the output of a previous layer. __Note that__ in original macro search space each square layer could choose any number of skip connections, while in our implementation it is only allowed to choose 0 or 1.

The result is shown in figure below (with the experiment config [here](https://github.com/microsoft/nni/blob/master/examples/trials/nas_cifar10/config_ppo.yml)):

![](../../img/ppo_cifar10.png)
