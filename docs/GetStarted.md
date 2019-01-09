**Get Started with NNI**
===

## **Installation**

* __Dependencies__

  ```bash
  python >= 3.5
  git
  wget
  ```

  python pip should also be correctly installed. You could use "python3 -m pip -v" to check in Linux.

  * Note: we don't support virtual environment in current releases.

* __Install NNI through pip__

  ```bash
  python3 -m pip install --user --upgrade nni
  ```

* __Install NNI through source code__

  ```bash
  git clone -b v0.4.1 https://github.com/Microsoft/nni.git
  cd nni
  source install.sh
  ```

## **Quick start: run a customized experiment**

An experiment is to run multiple trial jobs, each trial job tries a configuration which includes a specific neural architecture (or model) and hyper-parameter values. To run an experiment through NNI, you should:

* Provide a runnable trial
* Provide or choose a tuner
* Provide a yaml experiment configure file
* (optional) Provide or choose an assessor

**Prepare trial**: Let's use a simple trial example, e.g. mnist, provided by NNI. After you installed NNI, NNI examples have been put in ~/nni/examples, run `ls ~/nni/examples/trials` to see all the trial examples. You can simply execute the following command to run the NNI mnist example: 

```bash
python3 ~/nni/examples/trials/mnist-annotation/mnist.py
```

This command will be filled in the yaml configure file below. Please refer to [here](howto_1_WriteTrial.md) for how to write your own trial.

**Prepare tuner**: NNI supports several popular automl algorithms, including Random Search, Tree of Parzen Estimators (TPE), Evolution algorithm etc. Users can write their own tuner (refer to [here](howto_2_CustomizedTuner.md), but for simplicity, here we choose a tuner provided by NNI as below:

```yaml
tuner:
  builtinTunerName: TPE
    classArgs:
      optimize_mode: maximize
```

*builtinTunerName* is used to specify a tuner in NNI, *classArgs* are the arguments pass to the tuner, *optimization_mode* is to indicate whether you want to maximize or minimize your trial's result.

**Prepare configure file**: Since you have already known which trial code you are going to run and which tuner you are going to use, it is time to prepare the yaml configure file. NNI provides a demo configure file for each trial example, `cat ~/nni/examples/trials/mnist-annotation/config.yml` to see it. Its content is basically shown below:

```yaml
authorName: your_name
experimentName: auto_mnist

# how many trials could be concurrently running
trialConcurrency: 2

# maximum experiment running duration
maxExecDuration: 3h

# empty means never stop
maxTrialNum: 100

# choice: local, remote, pai
trainingServicePlatform: local

# choice: true, false  
useAnnotation: true
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python mnist.py
  codeDir: ~/nni/examples/trials/mnist-annotation
  gpuNum: 0
```

Here *useAnnotation* is true because this trial example uses our python annotation (refer to [here](../tools/annotation/README.md) for details). For trial, we should provide *trialCommand* which is the command to run the trial, provide *trialCodeDir* where the trial code is. The command will be executed in this directory. We should also provide how many GPUs a trial requires.

With all these steps done, we can run the experiment with the following command:

      nnictl create --config ~/nni/examples/trials/mnist-annotation/config.yml

You can refer to [here](NNICTLDOC.md) for more usage guide of *nnictl* command line tool.

## View experiment results
The experiment has been running now, NNI provides WebUI for you to view experiment progress, to control your experiment, and some other appealing features. The WebUI is opened by default by `nnictl create`.

## Read more

* [Tuners supported in the latest NNI release](./HowToChooseTuner.md)
* [Overview](Overview.md)
* [Installation](Installation.md)
* [Use command line tool nnictl](NNICTLDOC.md)
* [Use NNIBoard](WebUI.md)
* [Define search space](SearchSpaceSpec.md)
* [Config an experiment](ExperimentConfig.md)
* [How to run an experiment on local (with multiple GPUs)?](tutorial_1_CR_exp_local_api.md)
* [How to run an experiment on multiple machines?](tutorial_2_RemoteMachineMode.md)
* [How to run an experiment on OpenPAI?](PAIMode.md)
