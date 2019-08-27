PPO Tuner on NNI
===

## PPOTuner

This is a tuner generally for NNI's NAS interface, it uses ppo algorithm. The implementation inherits the main logic of the implementation here (i.e., ppo2 from openai), and is adapted for NAS scenario.

It could successfully tune the mnist-nas example, and has the following result:

![](../../img/ppo_mnist.png)

result of tuning enas search space (limited epoch number):

![](../../img/ppo_cifar10.png)