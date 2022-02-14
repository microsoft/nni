Customize Assessor
==================

NNI supports to build an assessor by yourself for tuning demand.

If you want to implement a customized Assessor, there are three things to do:


#. Inherit the base Assessor class
#. Implement assess_trial function
#. Configure your customized Assessor in experiment YAML config file

**1. Inherit the base Assessor class**

.. code-block:: python

   from nni.assessor import Assessor

   class CustomizedAssessor(Assessor):
       def __init__(self, ...):
           ...

**2. Implement assess trial function**

.. code-block:: python

   from nni.assessor import Assessor, AssessResult

   class CustomizedAssessor(Assessor):
       def __init__(self, ...):
           ...

       def assess_trial(self, trial_history):
           """
           Determines whether a trial should be killed. Must override.
           trial_history: a list of intermediate result objects.
           Returns AssessResult.Good or AssessResult.Bad.
           """
           # you code implement here.
           ...

**3. Configure your customized Assessor in experiment YAML config file**

NNI needs to locate your customized Assessor class and instantiate the class, so you need to specify the location of the customized Assessor class and pass literal values as parameters to the __init__ constructor.

.. code-block:: yaml

   assessor:
     codeDir: /home/abc/myassessor
     classFileName: my_customized_assessor.py
     className: CustomizedAssessor
     # Any parameter need to pass to your Assessor class __init__ constructor
     # can be specified in this optional classArgs field, for example
     classArgs:
       arg1: value1

Please noted in **2**. The object ``trial_history`` are exact the object that Trial send to Assessor by using SDK ``report_intermediate_result`` function.

The working directory of your assessor is ``<home>/nni-experiments/<experiment_id>/log``\ , which can be retrieved with environment variable ``NNI_LOG_DIRECTORY``\ ,

More detail example you could see:

* :githublink:`medianstop-assessor <nni/algorithms/hpo/medianstop_assessor.py>`
* :githublink:`curvefitting-assessor <nni/algorithms/hpo/curvefitting_assessor/>`

