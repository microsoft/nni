# **How To** - Customize Your Own Advisor

*Warning: API is subject to change in future releases.*

Advisor targets the scenario that the automl algorithm wants the methods of both tuner and assessor. Advisor is similar to tuner on that it receives trial parameters request, final results, and generate trial parameters. Also, it is similar to assessor on that it receives intermediate results, trial's end state, and could send trial kill command. Note that, if you use Advisor, tuner and assessor are not allowed to be used at the same time.

If a user want to implement a customized Advisor, she/he only needs to:

**1. Define an Advisor inheriting from the MsgDispatcherBase class.** For example:

```python
from nni.runtime.msg_dispatcher_base import MsgDispatcherBase

class CustomizedAdvisor(MsgDispatcherBase):
    def __init__(self, ...):
        ...
```

**2. Implement the methods with prefix `handle_` except `handle_request`**.. You might find [docs](https://nni.readthedocs.io/en/latest/sdk_reference.html#nni.runtime.msg_dispatcher_base.MsgDispatcherBase) for `MsgDispatcherBase` helpful.

**3. Configure your customized Advisor in experiment YAML config file.**

Similar to tuner and assessor. NNI needs to locate your customized Advisor class and instantiate the class, so you need to specify the location of the customized Advisor class and pass literal values as parameters to the `__init__` constructor.

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

**Note that** The working directory of your advisor is `<home>/nni-experiments/<experiment_id>/log`, which can be retrieved with environment variable `NNI_LOG_DIRECTORY`.

## Example

Here we provide an [example](https://github.com/microsoft/nni/tree/v1.9/examples/tuners/mnist_keras_customized_advisor).
