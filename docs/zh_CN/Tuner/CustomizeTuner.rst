自定义 Tuner
===============

自定义 Tuner
---------------

NNI 在内置的 Tuner 中提供了最新的调优算法。 NNI 同时也支持自定义 Tuner。

通过自定义 Tuner，可实现自己的调优算法。主要有三步：


#. 继承 Tuner 基类
#. 实现 receive_trial_result, generate_parameter 和 update_search_space 函数
#. 在 Experiment 的 YAML 文件中配置好自定义的 Tuner

示例如下：

**1. 继承 Tuner 基类**

.. code-block:: python

   from nni.tuner import Tuner

   class CustomizedTuner(Tuner):
       def __init__(self, ...):
           ...

**2. 实现 receive_trial_result, generate_parameter 和 update_search_space 函数**

.. code-block:: python

   from nni.tuner import Tuner

   class CustomizedTuner(Tuner):
       def __init__(self, ...):
           ...

       def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
           '''
           最终结果
           parameter_id: int
           parameters: 对象可由 'generate_parameters()' 创建
           value: Trial 最终指标，包括默认指标
           '''
           # 你的代码
       ...

       def generate_parameters(self, parameter_id, **kwargs):
           '''
           以可序列化对象的形式返回一组 Trial (超)参数
           parameter_id: int
           '''
           # 你的代码
           return your_parameters
       ...

       def update_search_space(self, search_space):
           '''
           Tuner 支持在运行时更新搜索空间。
           如果 Tuner 只在生成第一个超参前设置搜索空间，
           需要将其行为写到文档里。
           search_space: 定义 Experiment 时创建的 JSON 对象
           '''
           # 你的代码
       ...

``receive_trial_result`` 从输入中会接收 ``parameter_id, parameters, value`` 参数。 Tuner 会收到 Trial 进程发送的完全一样的 ``value`` 值。

``generate_parameters`` 函数返回的 ``your_parameters``，会被 NNI SDK 打包为 json。 然后 SDK 会将 json 对象解包给 Trial 进程。因此，Trial 进程会收到来自 Tuner 的完全相同的 ``your_parameters``。

例如：
如下实现了 ``generate_parameters``：

.. code-block:: python

   def generate_parameters(self, parameter_id, **kwargs):
       '''
       以可序列化对象的形式返回一组 Trial (超)参数
       parameter_id: int
       '''
       # 你的代码
       return {"dropout": 0.3, "learning_rate": 0.4}

这表示 Tuner 会一直生成超参组合 ``{"dropout": 0.3, "learning_rate": 0.4}``。 而 Trial 进程也会在调用 API ``nni.get_next_parameter()`` 时得到 ``{"dropout": 0.3, "learning_rate": 0.4}``。 Trial 结束后的返回值（通常是某个指标），通过调用 API ``nni.report_final_result()`` 返回给 Tuner。如： ``nni.report_final_result(0.93)``。 而 Tuner 的 ``receive_trial_result`` 函数会收到如下结果：

.. code-block:: python

   parameter_id = 82347
   parameters = {"dropout": 0.3, "learning_rate": 0.4}
   value = 0.93

**注意** ：Tuner 的工作目录是 ``<home>/nni-experiments/<experiment_id>/log``，可使用环境变量 ``NNI_LOG_DIRECTORY``，因此 ，如果要访问自己 Tuner 目录中的文件（如： ``data.txt``）不能直接使用 ``open('data.txt', 'r')``。 要使用：

.. code-block:: python

   _pwd = os.path.dirname(__file__)
   _fd = open(os.path.join(_pwd, 'data.txt'), 'r')

这是因为自定义的 Tuner 不是在自己的目录里执行的。（即，``pwd`` 返回的目录不是 Tuner 的目录）。

**3. 在 Experiment 的 YAML 文件中配置好自定义的 Tuner**

NNI 需要定位到自定义的 Tuner 类，并实例化它，因此需要指定自定义 Tuner 类的文件位置，并将参数值传给 ``__init__`` 构造函数。

.. code-block:: yaml

   tuner:
     codeDir: /home/abc/mytuner
     classFileName: my_customized_tuner.py
     className: CustomizedTuner
     # 所有的参数都需要传递给你 Assessor 的构造函数 __init__
     # 例如，可以在可选的 classArgs 字段中指定
     classArgs:
       arg1: value1

更多示例，可参考：

..

   * :githublink:`evolution-tuner <src/sdk/pynni/nni/evolution_tuner>`
   * :githublink:`hyperopt-tuner <src/sdk/pynni/nni/hyperopt_tuner>`
   * :githublink:`evolution-based-customized-tuner <examples/tuners/ga_customer_tuner>`


实现更高级的自动机器学习算法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上述内容足够写出通用的 Tuner。 但有时可能需要更多的信息，例如，中间结果， Trial 的状态等等，从而能够实现更强大的自动机器学习算法。 因此，有另一个 ``Advisor`` 类，直接继承于 ``MsgDispatcherBase``，它在 :githublink:`src/sdk/pynni/nni/msg_dispatcher_base.py <src/sdk/pynni/nni/msg_dispatcher_base.py>` 。 参考 `这里 <CustomizeAdvisor.rst>`__ 来了解如何实现自定义的 Advisor。
