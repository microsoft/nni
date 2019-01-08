# Customize-Tuner

## Customize Tuner

If you want to implement and use your own tuning algorithm, you can implement a customized Tuner, there are three things for you to do:

1) Inherit a tuner of a base Tuner class
2) Implement receive_trial_result and generate_parameter function
3) Configure your customized tuner in experiment yaml config file

Here is an example:

**1. Inherit a tuner of a base Tuner class**

```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
```

**2. Implement receive_trial_result and generate_parameter function**

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

**3. Configure your customized tuner in experiment yaml config file**

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

### Write a more advanced automl algorithm

The methods above are usually enough to write a general tuner. However, users may also want more methods, for example, intermediate results, trials' state (e.g., the methods in assessor), in order to have a more powerful automl algorithm. Therefore, we have another concept called `advisor` which directly inherits from `MsgDispatcherBase` in [`src/sdk/pynni/nni/msg_dispatcher_base.py`](../src/sdk/pynni/nni/msg_dispatcher_base.py). Please refer to [here](./howto_3_CustomizedAdvisor.md) for how to write a customized advisor.

## More advanced Tuner -- Advisor

Now let's talk about a special Tuner: Adivor. **The concept of adivisor is the advanced feature of tuner, skip this section will not affect using NNI.**

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

If you want to implement a customized Advisor, please refor to [How To - Customize Your Own Advisor][1]

[1]: https://github.com/Microsoft/nni/blob/master/docs/howto_3_CustomizedAdvisor.md