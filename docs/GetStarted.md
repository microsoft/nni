**Getting Started with NNI**
===

## **Installation**
* __Dependencies__

      python >= 3.5

    python pip should also be correctly installed. You could use "which pip" or "pip -V" to check in Linux.
    
    * Note: For now, we don't support virtual environment.

* __Install NNI through pip__

      pip3 install -v --user git+https://github.com/Microsoft/nni.git@v0.1
      source ~/.bashrc

* __Install NNI through source code__
   
      git clone -b v0.1 https://github.com/Microsoft/nni.git
      cd nni
      chmod +x install.sh
      source install.sh


## **Quick start: run a customized experiment**
An experiment is to run multiple trial jobs, each trial job tries a configuration which includes a specific neural architecture (or model) and hyper-parameter values. To run an experiment through NNI, you should:

* Provide a runnable trial
* Provide or choose a tuner
* Provide a yaml experiment configure file
* (optional) Provide or choose an assessor

**Prepare trial**: Let's use a simple trial example, e.g. mnist, provided by NNI. After you installed NNI, NNI examples have been put in ~/nni/examples, run `ls ~/nni/examples/trials` to see all the trial examples. You can simply execute the following command to run the NNI mnist example: 

      python ~/nni/examples/trials/mnist-annotation/mnist.py

This command will be filled in the yaml configure file below. Please refer to [here]() for how to write your own trial.

**Prepare tuner**: NNI supports several popular automl algorithms, including Random Search, Tree of Parzen Estimators (TPE), Evolution algorithm etc. Users can write their own tuner (refer to [here](CustomizedTuner.md)), but for simplicity, here we choose a tuner provided by NNI as below:

      tuner:
        builtinTunerName: TPE
        classArgs:
          optimize_mode: maximize

*builtinTunerName* is used to specify a tuner in NNI, *classArgs* are the arguments pass to the tuner (the spec of builtin tuners can be found [here]()), *optimization_mode* is to indicate whether you want to maximize or minimize your trial's result.

**Prepare configure file**: Since you have already known which trial code you are going to run and which tuner you are going to use, it is time to prepare the yaml configure file. NNI provides a demo configure file for each trial example, `cat ~/nni/examples/trials/mnist-annotation/config.yml` to see it. Its content is basically shown below:

```
authorName: your_name
experimentName: auto_mnist

# how many trials could be concurrently running
trialConcurrency: 2

# maximum experiment running duration
maxExecDuration: 3h

# empty means never stop
maxTrialNum: 100

# choice: local, remote  
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

## Further reading
* [How to write a trial running on NNI (Mnist as an example)?](WriteYourTrial.md)
* [Tutorial of NNI python annotation.](../tools/nni_annotation/README.md)
* [Tuners supported by NNI.](../src/sdk/pynni/nni/README.md)
* [How to enable early stop (i.e. assessor) in an experiment?](EnableAssessor.md)
* [How to run an experiment on multiple machines?](RemoteMachineMode.md)
* [How to write a customized tuner?](CustomizedTuner.md)
* [How to write a customized assessor?](../examples/assessors/README.md)
* [How to resume an experiment?](NNICTLDOC.md)
* [Tutorial of the command tool *nnictl*.](NNICTLDOC.md)
