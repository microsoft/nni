PPO Tuner on NNI
===

## PPOTuner

This is a tuner generally for NNI's NAS interface, it uses ppo algorithm. The implementation inherits the main logic of the implementation here (i.e., ppo2 from openai), and is adapted for NAS scenario.

It could successfully tune the mnist-nas example, and has the following result:

![](../../img/bohb_1.png)

result of tuning enas search space (limited epoch number):

![](../../img/bohb_1.png)