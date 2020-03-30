PBT Tuner on NNI
===

## PBTTuner

Population Based Training (PBT) comes from [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846v1). It's a simple asynchronous optimization algorithm which effectively utilizes a fixed computational budget to jointly optimize a population of models and their hyperparameters to maximize performance. Importantly, PBT discovers a schedule of hyperparameter settings rather than following the generally sub-optimal strategy of trying to find a single fixed set to use for the whole course of training.

PBTTuner initializes a population with several trials. Users can set a specific number of training epochs. After a certain number of epochs, the parameters and hyperparameters in the trial with bad metrics will be replaced with a better trial (exploit). Then the hyperparameters are perturbed (explore).

In our implementation, training epochs in the trial code is regarded as a step of PBT, different with other tuners. At the end of each step, PBT tuner will do exploitation and exploration -- replacing some trials with new trials. This is implemented by constantly modifying the values of `load_checkpoint_dir` and `save_checkpoint_dir`. We can directly change `load_checkpoint_dir` to replace parameters and hyperparameters, and `save_checkpoint_dir` to save a checkpoint that will be loaded in the next step. To this end, we need a shared folder which is accessible to all trials.

If the experiment is running in local mode, users could provide an argument `all_checkpoint_dir` which will be the base folder of `load_checkpoint_dir` and `save_checkpoint_dir` (`checkpoint_dir` is set to `all_checkpoint_dir/<population-id>/<step>`). By default, `all_checkpoint_dir` is set to be `~/nni/experiments/<exp-id>/checkpoint`. If the experiment is in non-local mode, then users should provide a path in a shared storage folder which is mounted at `all_checkpoint_dir` on worker machines (but it's not necessarily available on the machine which runs tuner).
