NNI 中的 PBTTuner
===

## PBTTuner

Population Based Training (PBT，基于种群的训练) 来自于 [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846v1)。 它是一种简单的异步优化算法，在固定的计算资源下，它能有效的联合优化一组模型及其超参来最大化性能。 Importantly, PBT discovers a schedule of hyperparameter settings rather than following the generally sub-optimal strategy of trying to find a single fixed set to use for the whole course of training.

PBTTuner 通过几次 Trial 来初始化种群。 用户可以设置特定数量的训练 Epoch。 After a certain number of epochs, the parameters and hyperparameters in the trial with bad metrics will be replaced with a better trial (exploit). Then the hyperparameters are perturbed (explore).

In our implementation, training epochs in the trial code is regarded as a step of PBT, different with other tuners. At the end of each step, PBT tuner will do exploitation and exploration -- replacing some trials with new trials. This is implemented by constantly modifying the values of `load_checkpoint_dir` and `save_checkpoint_dir`. We can directly change `load_checkpoint_dir` to replace parameters and hyperparameters, and `save_checkpoint_dir` to save a checkpoint that will be loaded in the next step. To this end, we need a shared folder which is accessible to all trials.

If the experiment is running in local mode, users could provide an argument `all_checkpoint_dir` which will be the base folder of `load_checkpoint_dir` and `save_checkpoint_dir` (`checkpoint_dir` is set to `all_checkpoint_dir/<population-id>/<step>`). By default, `all_checkpoint_dir` is set to be `~/nni/experiments/<exp-id>/checkpoint`. If the experiment is in non-local mode, then users should provide a path in a shared storage folder which is mounted at `all_checkpoint_dir` on worker machines (but it's not necessarily available on the machine which runs tuner).
