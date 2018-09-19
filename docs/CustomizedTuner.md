# Customized Tuner for Experts

*Tuner receive result from Trial as a matric to evaluate the performance of a specific parameters/architecture configure. And tuner send next hyper-parameter or architecture configure to Trial.*

So, if user want to implement a customized Tuner, she/he only need to:

1) Inherit a tuner of a base Tuner class
2) Implement receive_trial_result and generate_parameter function
3) Write a script to run Tuner

Here ia an example:

**1) Inherit a tuner of a base Tuner class**
```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
```

**2) Implement receive_trial_result and generate_parameter function**
```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
    
    def receive_trial_result(self, parameter_id, parameters, reward):
    '''
    Record an observation of the objective function and Train
    parameter_id: int
    parameters: object created by 'generate_parameters()'
    reward: object reported by trial
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
```receive_trial_result``` will receive ```the parameter_id, parameters, reward``` as parameters input. Also, Tuner will receive the ```reward``` object are exactly same reward that Trial send.

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
It's means your Tuner will always generate parameters ```{"dropout": 0.3, "learning_rate": 0.4}```. Then Trial will receive ```{"dropout": 0.3, "learning_rate": 0.4}``` this object will using ```nni.get_parameters()``` API from NNI SDK. After training of Trial, it will send result to Tuner by calling ```nni.report_final_result(0.93)```. Then ```receive_trial_result``` will function will receied these parameters like：
```
parameter_id = 82347
parameters = {"dropout": 0.3, "learning_rate": 0.4}
reward = 0.93
```

**Note that** if you want to access a file (e.g., ```data.txt```) in the directory of your own tuner, you cannot use ```open('data.txt', 'r')```. Instead, you should use the following:
```
_pwd = os.path.dirname(__file__)
_fd = open(os.path.join(_pwd, 'data.txt'), 'r')
```
This is because your tuner is not executed in the directory of your tuner (i.e., ```pwd``` is not the directory of your own tuner).

**3) Configure your customized tuner in experiment yaml config file**

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
