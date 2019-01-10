# **指南** - 自定义调参器

*调参器从尝试接收指标结果，来评估一组超参或网络结构的性能。 然后调参器会将下一组超参或网络结构的配置发送给新的尝试。*

因此，如果要自定义调参器，需要：

1) Inherit a tuner of a base Tuner class 2) Implement receive_trial_result and generate_parameter function 3) Configure your customized tuner in experiment yaml config file

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

    receive_trial_result will receive ```the parameter_id, parameters, value``` as parameters input. Also, Tuner will receive the ```value``` object are exactly same value that Trial send.

The ```your_parameters``` return from ```generate_parameters``` function, will be package as json object by NNI SDK. NNI SDK will unpack json object so the Trial will receive the exact same ```your_parameters``` from Tuner.

For example: If the you implement the ```generate_parameters``` like this:

```python
    def generate_parameters(self, parameter_id):
        '''
        返回尝试的超参组合的序列化对象
        parameter_id: int
        '''
        # 代码实现位置
        return {"dropout": 0.3, "learning_rate": 0.4}
```

It means your Tuner will always generate parameters ```{"dropout": 0.3, "learning_rate": 0.4}```. Then Trial will receive ```{"dropout": 0.3, "learning_rate": 0.4}``` by calling API ```nni.get_next_parameter()```. Once the trial ends with a result (normally some kind of metrics), it can send the result to Tuner by calling API ```nni.report_final_result()```, for example ```nni.report_final_result(0.93)```. Then your Tuner's ```receive_trial_result``` function will receied the result like：

    parameter_id = 82347
    parameters = {"dropout": 0.3, "learning_rate": 0.4}
    value = 0.93
    

**Note that** if you want to access a file (e.g., ```data.txt```) in the directory of your own tuner, you cannot use ```open('data.txt', 'r')```. Instead, you should use the following:

    _pwd = os.path.dirname(__file__)
    _fd = open(os.path.join(_pwd, 'data.txt'), 'r')
    

This is because your tuner is not executed in the directory of your tuner (i.e., ```pwd``` is not the directory of your own tuner).

**3) Configure your customized tuner in experiment yaml config file**

NNI needs to locate your customized tuner class and instantiate the class, so you need to specify the location of the customized tuner class and pass literal values as parameters to the \_\_init__ constructor.

```yaml
tuner:
  codeDir: /home/abc/mytuner
  classFileName: my_customized_tuner.py
  className: CustomizedTuner
  # 任何传入 __init__ 构造函数的参数
  # 都需要声明在 classArgs 字段中，如：
  classArgs:
    arg1: value1
```

More detail example you could see:

> - [evolution-tuner](../../src/sdk/pynni/nni/evolution_tuner)
> - [hyperopt-tuner](../../src/sdk/pynni/nni/hyperopt_tuner)
> - [evolution-based-customized-tuner](../../examples/tuners/ga_customer_tuner)

## 实现更高级的自动机器学习算法

The methods above are usually enough to write a general tuner. However, users may also want more methods, for example, intermediate results, trials' state (e.g., the methods in assessor), in order to have a more powerful automl algorithm. Therefore, we have another concept called `advisor` which directly inherits from `MsgDispatcherBase` in [`src/sdk/pynni/nni/msg_dispatcher_base.py`](../src/sdk/pynni/nni/msg_dispatcher_base.py). Please refer to [here](./howto_3_CustomizedAdvisor.md) for how to write a customized advisor.