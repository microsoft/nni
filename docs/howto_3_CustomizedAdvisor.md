# **How To** - Customize Your Own Advisor

*Advisor targets the scenario that the automl algorithm wants the methods of both tuner and assessor. Advisor is similar to tuner on that it receives trial parameters request, final results, and generate trial parameters. Also, it is similar to assessor on that it receives intermediate results, trial's end state, and could send trial kill command. Note that, if you use Advisor, tuner and assessor are not allowed to be used at the same time.*

So, if user want to implement a customized Advisor, she/he only need to:

1. Define an Advisor inheriting from the MsgDispatcherBase class
1. Implement the methods with prefix `handle_` except `handle_request`
1. Configure your customized Advisor in experiment yaml config file

Here is an example:

**1) Define an Advisor inheriting from the MsgDispatcherBase class**

```python
from nni.msg_dispatcher_base import MsgDispatcherBase

class CustomizedAdvisor(MsgDispatcherBase):
    def __init__(self, ...):
        ...
```

**2) Implement the methods with prefix `handle_` except `handle_request`**

Please refer to the implementation of Hyperband ([src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py](../src/sdk/pynni/nni/hyperband_advisor/hyperband_advisor.py)) for how to implement the methods.

**3) Configure your customized Advisor in experiment yaml config file**

Similar to tuner and assessor. NNI needs to locate your customized Advisor class and instantiate the class, so you need to specify the location of the customized Advisor class and pass literal values as parameters to the \_\_init__ constructor.

```yaml
advisor:
  codeDir: /home/abc/myadvisor
  classFileName: my_customized_advisor.py
  className: CustomizedAdvisor
  # Any parameter need to pass to your advisor class __init__ constructor
  # can be specified in this optional classArgs field, for example 
  classArgs:
    arg1: value1
```
