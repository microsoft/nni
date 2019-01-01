# QuickStart

## Installation

Two ways for installation: you can get NNI from pypi or clone the repo and install it from there.

**Install through pip**

* We support Linux and MacOS in current stage, Ubuntu 16.04 or higher, along with MacOS 10.14.1 are tested and supported. Simply run the following `pip install` in an environment that has `python >= 3.5`.

```bash
    python3 -m pip install --user --upgrade nni
```

* Note:
  * If you are in docker container (as root), please remove `--user` from the installation command.
  * If there is any error like `Segmentation fault`, please refer to [FAQ][1]

**Install through source code**

* We support Linux (Ubuntu 16.04 or higher), MacOS (10.14.1) in our current stage. 
* Run the following commands in an environment that has `python >= 3.5`, `git` and `wget`.

```bash
    git clone -b v0.4.1 https://github.com/Microsoft/nni.git
    cd nni
    source install.sh
```

For the system requirements of NNI, please refer to [Install NNI][2]

## First "Hello World" experiment: MNIST

Now let's start to run our first MINIST experiment using NNI.

Here is a configuration file in an NNI experiment:

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
trainingServicePlatform: local
searchSpacePath: search_space.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
```

**We did two things here:**

* specify the `range` of hyper-paramaters to be searched for this experiment in `search_space.json`.
* define a NNI trial in `mnist.py`, an individual attempt at applying a set of parameters on a model.

You can [click here][3] to find the example of `config.yml`, `search_space.json` and `mnist.py` in this example.

Note: Other configurations can be kept by their default value. We will talk about them in the following tutorial.

* Everything is ready now! The experiment can be run from the command-line:

```bash
    nnictl create --config nni/examples/trials/mnist/config.yml
```

* Wait for the message `INFO: Successfully started experiment!` in the command line. This message indicates that your experiment has been successfully started. And this is what we expected to get:

```
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
-----------------------------------------------------------------------
The experiment id is egchD4qy
The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
-----------------------------------------------------------------------

You can use these commands to get more information about the experiment
-----------------------------------------------------------------------
         commands                       description
1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl log stderr             show stderr log content
4. nnictl log stdout             show stdout log content
5. nnictl stop                   stop an experiment
6. nnictl trial kill             kill a trial job by id
7. nnictl --help                 get help information about nnictl
-----------------------------------------------------------------------
```

## WebUI

After you start your experiment in NNI successfully, you can find a message in the command-line interface to tell you `Web UI url` like this:

```
The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
```

Open the `Web UI url` in your browser, you can view detail information of the experiment and all the submitted trial jobs as shown below.

### View summary page

Click the tab "Overview".

* See the experiment trial profile and search space message.
* Support to download the experiment result.

![](./img/over1.png)
* See good performance trials.

![](./img/over2.png)

### View job default metric

Click the tab "Default Metric" to see the point graph of all trials. Hover to see its specific default metric and search space message.

![](./img/accuracy.png)

### View hyper parameter

Click the tab "Hyper Parameter" to see the parallel graph.

* You can select the percentage to see top trials.
* Choose two axis to swap its positions

![](./img/hyperPara.png)

### View Trial Duration

Click the tab "Trial Duration" to see the bar graph.

![](./img/trial_duration.png)

### View trials status 

Click the tab "Trials Detail" to see the status of the all trials. Specifically:

* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy and search space file.
* If you run a pai experiment, you can also see the hdfsLogPath.

![](./img/table_openrow.png)

* Kill: you can kill a job that status is running.
* Support to search for a specific trial.
* Intermediate Result Graph.

![](./img/intermediate.png)

[1]: https://github.com/Microsoft/nni/blob/master/docs/FAQ.md
[2]: https://github.com/Microsoft/nni/blob/master/docs/Installation.md
[3]: https://github.com/Microsoft/nni/tree/master/examples/trials/mnist