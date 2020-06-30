**Run an Experiment on Azure Machine Learning**
===
NNI supports running an experiment on [AML](https://azure.microsoft.com/en-us/services/machine-learning/) , called aml mode.

## Setup environment
Step 1. Install NNI, follow the install guide [here](../Tutorial/QuickStart.md).   

Step 2. Create AML account, follow the document [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli).

Step 3. Get your account information.
![](../../img/aml_account.png)

Step4. Install AML package environment.
```
python3 -m pip install azureml --user
python3 -m pip install azureml-sdk --user
```

## Run an experiment
Use `examples/trials/mnist-tfv1` as an example. The NNI config YAML file's content is like:

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
trainingServicePlatform: aml
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  computerTarget: ussc40rscl
  nodeCount: 1
amlConfig:
  subscriptionId: ${replace_to_your_subscriptionId}
  resourceGroup: ${replace_to_your_resourceGroup}
  workspaceName: ${replace_to_your_workspaceName}

```

Note: You should set `trainingServicePlatform: aml` in NNI config YAML file if you want to start experiment in aml mode.

Compared with [LocalMode](LocalMode.md) trial configuration in aml mode have these additional keys:
* computerTarget
    * required key. The computer cluster name you want to use in your AML workspace.
* nodeCount
    * required key. The node count each run in your experiment.

amlConfig:
* subscriptionId
    * the subscriptionId of your account
* resourceGroup
    * the resourceGroup of your account
* workspaceName
    * the workspaceName of your account
  