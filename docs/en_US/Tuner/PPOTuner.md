PPO Tuner on NNI
===

## PPOTuner

This is a tuner generally for NNI's NAS interface, it uses [ppo algorithm](https://arxiv.org/abs/1707.06347). The implementation inherits the main logic of the implementation [here](https://github.com/openai/baselines/tree/master/baselines/ppo2) (i.e., ppo2 from openai), and is adapted for NAS scenario.

It could successfully tune the [mnist-nas example](https://github.com/microsoft/nni/tree/master/examples/trials/mnist-nas), and has the following result:

![](../../img/ppo_mnist.png)

result of tuning [enas search space](https://github.com/microsoft/nni/tree/master/examples/trials/nas_cifar10) (limited epoch number):

![](../../img/ppo_cifar10.png)
