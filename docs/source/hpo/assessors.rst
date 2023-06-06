Assessor: Early Stopping
========================

In HPO, some hyperparameter sets may have obviously poor performance and it will be unnecessary to finish the evaluation.
This is called *early stopping*, and in NNI early stopping algorithms are called *assessors*.

An assessor monitors *intermediate results* of each *trial*.
If a trial is predicted to produce suboptimal final result, the assessor will stop that trial immediately,
to save computing resources for other hyperparameter sets.

As introduced in quickstart tutorial, a trial is the evaluation process of a hyperparameter set,
and intermediate results are reported with :func:`nni.report_intermediate_result` API in trial code.
Typically, intermediate results are accuracy or loss metrics of each epoch.

Using an assessor will increase the efficiency of computing resources,
but may slightly reduce the predicition accuracy of tuners.
It is recommended to use an assessor when computing resources are insufficient.

Common Usage
------------

The usage of assessors are similar to tuners.

To use a built-in assessor you need to specify its name and arguments:

.. code-block:: python

    config.assessor.name = 'Medianstop'
    config.assessor.class_args = {'optimize_mode': 'maximize'}

Built-in Assessors
------------------

.. list-table::
    :header-rows: 1
    :widths: auto
 
    * - Assessor
      - Brief Introduction of Algorithm
 
    * - :class:`Median Stop <nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor>`
      - Stop if the hyperparameter set performs worse than median at any step.

    * - :class:`Curve Fitting <nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor>`
      - Stop if the learning curve will likely converge to suboptimal result.
