自定义 Assessor
==================

NNI 支持自定义 Assessor。

实现自定义的 Assessor，需要如下几步：


#. 继承 Assessor 基类
#. 实现 assess_trial 函数
#. 在 Experiment 的 YAML 文件中配置好自定义的 Assessor

**1. 继承 Assessor 基类**

.. code-block:: python

   from nni.assessor import Assessor

   class CustomizedAssessor(Assessor):
       def __init__(self, ...):
           ...

**2. 实现 assess_trial 函数**

.. code-block:: python

   from nni.assessor import Assessor, AssessResult

   class CustomizedAssessor(Assessor):
       def __init__(self, ...):
           ...

       def assess_trial(self, trial_history):
           """
           决定 Trial 是否应该被终止。 必须重载。
           trial_history: 中间结果列表对象。
           返回 AssessResult.Good 或 AssessResult.Bad。
           """
           # 你的代码
           ...

**3. 在 Experiment 的 YAML 文件中配置好自定义的 Assessor**

NNI 需要定位到自定义的 Assessor 类，并实例化它，因此需要指定自定义 Assessor 类的文件位置，并将参数值传给 __init__ 构造函数。

`论文 <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf>`__。 

.. code-block:: yaml

   assessor:
      codeDir: /home/abc/myassessor
      classFileName: my_customized_assessor.py
      className: CustomizedAssessor
      # 所有的参数都需要传递给你 Assessor 的构造函数 __init__
      # 例如，可以在可选的 classArgs 字段中指定
      classArgs:
        arg1: value1

注意 **2** 中： 对象 ``trial_history`` 和 ``report_intermediate_result`` 函数返回给 Assessor 的完全一致。

Assessor 的工作目录是 ``<home>/nni-experiments/<experiment_id>/log``\  ，可从环境变量 ``NNI_LOG_DIRECTORY``\ 中获取。

更多示例，可参考：

..

   * :githublink:`medianstop-assessor <src/sdk/pynni/nni/medianstop_assessor>`
   * :githublink:`curvefitting-assessor <src/sdk/pynni/nni/curvefitting_assessor>`

