# Tuners

## Overview

NNI provides an easy to adopt approach to set up parameter tuning algorithms, we call them **Tuners**. Tuner receive result from Trial as a matric to evaluate the performance of a specific parameters/architecture configure. And tuner send next hyper-parameter or architecture configure to Trial.

In NNI, we support two approaches to set the tuner.

1. Directly use tuner provided by nni sdk

        required fields: builtinTunerName and classArgs.

2. Customize your own tuner file

        required fields: codeDirectory, classFileName, className and classArgs.

For now, NNI has supported the following tuner algorithms:

|Tuner|Brief introduction to the algorithm|Suggested scenario|Usage|Reference|
|---|---|---|---|---|
|**TPE**|The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model.|TPE, as a black-box optimization, can be used in various scenarios, and shows good performance in general. Especially when you have limited computation resource and can only try a small number of trials. From a large amount of experiments, we could found that TPE is far better than Random Search.|[TPE Usage][1]|Paper: [Algorithms for Hyper-Parameter Optimization][2]|
|**Random Search**|In Random Search for Hyper-Parameter Optimization show that Random Search might be surprisingly simple and effective. We suggests that we could use Random Search as baseline when we have no knowledge about the prior distribution of hyper-parameters.|Random search is suggested when each trial does not take too long (e.g., each trial can be completed very soon, or early stopped by assessor quickly), and you have enough computation resource. Or you want to uniformly explore the search space. Random Search could be considered as baseline of search algorithm.|[Random Search Usage][3]||
|**Anneal**|This simple annealing algorithm begins by sampling from the prior, but tends over time to sample from points closer and closer to the best ones observed. This algorithm is a simple variation on random search that leverages smoothness in the response surface. The annealing rate is not adaptive.|Anneal is suggested when each trial does not take too long, and you have enough computation resource(almost same with Random Search). Or the variables in search space could be sample from some prior distribution.|[Anneal Usage][4]||
|**Naive Evolution**|Naive Evolution comes from Large-Scale Evolution of Image Classifiers. It randomly initializes a population based on search space. For each generation, it chooses better ones and do some mutation (e.g., change a hyperparameter, add/remove one layer) on them to get the next generation. Naive Evolution requires many trials to works, but it's very simple and easily to expand new features.|Its requirement of computation resource is relatively high. Specifically, it requires large inital population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress.|[Naive Evolution][5]|Paper: [Large-Scale Evolution of Image Classifiers][6]|
|**SMAC**(to install through `nnictl`)|SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO, in order to handle categorical parameters. The SMAC supported by nni is a wrapper on the SMAC3 github repo.|Similar to TPE, SMAC is also a black-box tuner which can be tried in various scenarios, and is suggested when computation resource is limited. It is optimized for discrete hyperparameters, thus, suggested when most of your hyperparameters are discrete.|[SMAC Usage][7]|Paper: [Sequential Model-Based Optimization for General Algorithm Configuration][8], Github repo: [automl/SMAC3][9]|
|**Batch tuner**|Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type choice in search space spec.|If the configurations you want to try have been decided, you can list them in searchspace file (using choice) and run them using batch tuner.|[Batch tuner Usage][10]||
|**Grid Search**|Grid Search performs an exhaustive searching through a manually specified subset of the hyperparameter space defined in the searchspace file. Note that the only acceptable types of search space are choice, quniform, qloguniform. The number q in quniform and qloguniform has special meaning (different from the spec in search space spec). It means the number of values that will be sampled evenly from the range low and high.|It is suggested when search space is small, it is feasible to exhaustively sweeping the whole search space.|[Grid Search Usage][11]||
|**Hyperband**|Hyperband tries to use limited resource to explore as many configurations as possible, and finds out the promising ones to get the final result. The basic idea is generating many configurations and to run them for small number of STEPs to find out promising one, then further training those promising ones to select several more promising one.|It is suggested when you have limited computation resource but have relatively large search space. It performs good in the scenario that intermediate result (e.g., accuracy) can reflect good or bad of final result (e.g., accuracy) to some extent.|[Hyperband Usage][12]|Paper: [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization][13]|
|**Network Morphism**|Network Morphism provides functions to automatically search for architecture of deep learning models. Every child network inherits the knowledge from its parent network and morphs into diverse types of networks, including changes of depth, width and skip-connection. Next, it estimates the value of child network using the history architecture and metric pairs. Then it selects the most promising one to train.|It is suggested that you want to apply deep learning methods to your task (your own dataset) but you have no idea of how to choose or design a network. You modify the example to fit your own dataset and your own data augmentation method. Also you can change the batch size, learning rate or optimizer. It is feasible for different tasks to find a good network architecture. Now this tuner only supports the cv domain.|[Network Morphism Usage][14]|Paper: [Auto-Keras: Efficient Neural Architecture Search with Network Morphism][15]|

## Try Different Tuners

### Example of Builtin Tuner Usage

Our NNI integrates state-of-the-art tuning algorithm. You can easily use our builtin tuner by declare the `builtinTunerName` and `classArguments` in config file.

For example, if you chose to use "TPE" tuner, you can set the `config.yml` like this:

```yaml
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, Grid Search, Hyperband, Network Morphism
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
```

If you chose to use "Anneal" tuner, you can set the `config.yml` like this:

```yaml
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, Grid Search, Hyperband, Network Morphism
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: Anneal
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
```

If you chose to use "Network Morphism" tuner, you can set the `config.yml` like this:

```yaml
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, Grid Search, Hyperband, Network Morphism
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: NetworkMorphism
  classArgs:
    optimize_mode: maximize
    task: cv
    input_width: 32
    input_channel: 3
    n_output_node: 10
```

Note:

* Please write the .yml file in this format correctly.
* For more examples Tuner usage, please refer to [reference][16] part.

### Requirements of each tuner

According to different usage scenarios and requirements, we encourage users to use different tuner to better adjust the hyper-parameters. The following lists the names of our current builtin tuner and the corresponding classArg.

Notes:

* **Required** means you must declare this classArg.
* **Unsupported** means we don't support this classArg.
* Other characters indicate that this classArg is **Optional**, we use their default value.

|Tuner|builtinTunerName|optimize_mode('maximize'/'minimize')|Other Optional classArg|
|---|---|---|---|
|**TPE**|TPE|'maximize'||
|**Random Search**|Random|'maximize'||
|**Anneal**|Anneal|'maximize'||
|**Naive Evolution**|Evolution|'maximize'||
|**SMAC**|SMAC|'maximize'||
|**Batch tuner**|Batch tuner|Unsupported||
|**Grid Search**|Grid Search|Unsupported||
|**Hyperband**|Hyperband|'maximize'|['R']: int, ['eta']: int|
|**Network Morphism**|NetworkMorphism|'maximize'|['task']: str('cv', 'nlp', 'common'), ['input_width']: int, ['input_channel']: int, ['n_output_node']: int|

## Customize Tuner

If you want to implement and use your own tuning algorithm, you can implement a customized Tuner, there are three things for you to do:

1) Inherit a tuner of a base Tuner class
2) Implement receive_trial_result and generate_parameter function
3) Configure your customized tuner in experiment yaml config file

Here ia an example:

**1.Inherit a tuner of a base Tuner class**

```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
```

**2.Implement receive_trial_result and generate_parameter function**

```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...

    def receive_trial_result(self, parameter_id, parameters, value):
    '''
    Record an observation of the objective function and Train
    parameter_id: int
    parameters: object created by 'generate_parameters()'
    value: final metrics of the trial, including reward
    '''
    # your code implements here.
    ...

    def generate_parameters(self, parameter_id):
    '''
    Returns a set of trial (hyper-)parameters, as a serializable object
    parameter_id: int
    '''
    # your code implements here.
    return your_parameters
    ...
```

```receive_trial_result``` will receive ```the parameter_id, parameters, value``` as parameters input. Also, Tuner will receive the ```value``` object are exactly same value that Trial send.

The ```your_parameters``` return from ```generate_parameters``` function, will be package as json object by NNI SDK. NNI SDK will unpack json object so the Trial will receive the exact same ```your_parameters``` from Tuner.

For example:
If the you implement the ```generate_parameters``` like this:

```python

def generate_parameters(self, parameter_id):
    '''
    Returns a set of trial (hyper-)parameters, as a serializable object
    parameter_id: int
    '''
    # your code implements here.
    return {"dropout": 0.3, "learning_rate": 0.4}

```

It means your Tuner will always generate parameters ```{"dropout": 0.3, "learning_rate": 0.4}```. Then Trial will receive ```{"dropout": 0.3, "learning_rate": 0.4}``` by calling API ```nni.get_next_parameter()```. Once the trial ends with a result (normally some kind of metrics), it can send the result to Tuner by calling API ```nni.report_final_result()```, for example ```nni.report_final_result(0.93)```. Then your Tuner's ```receive_trial_result``` function will receied the result like：

```

parameter_id = 82347
parameters = {"dropout": 0.3, "learning_rate": 0.4}
value = 0.93

```

**Note that** if you want to access a file (e.g., ```data.txt```) in the directory of your own tuner, you cannot use ```open('data.txt', 'r')```. Instead, you should use the following:

```

_pwd = os.path.dirname(__file__)
_fd = open(os.path.join(_pwd, 'data.txt'), 'r')

```

This is because your tuner is not executed in the directory of your tuner (i.e., ```pwd``` is not the directory of your own tuner).

**3.Configure your customized tuner in experiment yaml config file**

NNI needs to locate your customized tuner class and instantiate the class, so you need to specify the location of the customized tuner class and pass literal values as parameters to the \_\_init__ constructor.

```yaml

tuner:
  codeDir: /home/abc/mytuner
  classFileName: my_customized_tuner.py
  className: CustomizedTuner
  # Any parameter need to pass to your tuner class __init__ constructor
  # can be specified in this optional classArgs field, for example 
  classArgs:
    arg1: value1

```

More detail example you could see:
> * [evolution-tuner](../src/sdk/pynni/nni/evolution_tuner)
> * [hyperopt-tuner](../src/sdk/pynni/nni/hyperopt_tuner)
> * [evolution-based-customized-tuner](../examples/tuners/ga_customer_tuner)

## Write a more advanced automl algorithm

The methods above are usually enough to write a general tuner. However, users may also want more methods, for example, intermediate results, trials' state (e.g., the methods in assessor), in order to have a more powerful automl algorithm. Therefore, we have another concept called `advisor` which directly inherits from `MsgDispatcherBase` in [`src/sdk/pynni/nni/msg_dispatcher_base.py`](../src/sdk/pynni/nni/msg_dispatcher_base.py). Please refer to [here](./howto_3_CustomizedAdvisor.md) for how to write a customized advisor.

## More advanced Tuner -- Advisor

Now let's talk about a special Tuner: Adivor. **The concept of adivisor doesn't affect the usage of NNI, so you can skip this part if you are in a hurry.**

Advisor targets the scenario that the automl algorithm wants the methods of both tuner and assessor. Advisor is similar to tuner on that it receives trial configuration request, final results, and generate trial configurations. Also, it is similar to assessor on that it receives intermediate results, trial's end state, and could send trial kill command. Note that, if you use Advisor, tuner and assessor are not allowed to be used at the same time.

Currently NNI only have hyperband to use Advisor. Hyperband tries to use limited resource to explore as many configurations as possible, and finds out the promising ones to get the final result. The basic idea is generating many configurations and to run them for small number of STEPs to find out promising one, then further training those promising ones to select several more promising one.

Suggested scenario: Its requirement of computation resource is relatively high. Specifically, it requires large inital population to avoid falling into local optimum. If your trial is short or leverages assessor, this tuner is a good choice. And, it is more suggested when your trial code supports weight transfer, that is, the trial could inherit the converged weights from its parent(s). This can greatly speed up the training progress.

If you want to use hyperband, here is the usage example:

```yaml

  # config.yaml
  advisor:
    builtinAdvisorName: Hyperband
    classArgs:
      # choice: maximize, minimize
      optimize_mode: maximize
      # R: the maximum STEPS (could be the number of mini-batches or epochs) can be
      #    allocated to a trial. Each trial should use STEPS to control how long it runs.
      R: 60
      # eta: proportion of discarded trials
      eta: 3

```

If you want to implement a customized Advisor, please refor to [How To - Customize Your Own Advisor][17]

[1]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[2]: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
[3]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[4]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[5]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[6]: https://arxiv.org/pdf/1703.01041.pdf
[7]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[8]: https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
[9]: https://github.com/automl/SMAC3
[10]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[11]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[12]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[13]: https://arxiv.org/pdf/1603.06560.pdf
[14]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[15]: https://arxiv.org/pdf/1806.10282.pdf
[16]: https://github.com/Microsoft/nni/blob/master/docs/HowToChooseTuner.md
[17]: https://github.com/Microsoft/nni/blob/master/docs/howto_3_CustomizedAdvisor.md