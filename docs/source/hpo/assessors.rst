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
    config.tuner.class_args = {'optimize_mode': 'maximize'}

Built-in Assessors
------------------

.. list-table::
    :header-rows: 1
    :widths: auto
 
    * - Assessor
      - Brief Introduction of Algorithm
 
    * - :class:`Medianstop <nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor>`
      - It stops a pending trial X at step S if
        the trial’s best objective value by step S is strictly worse than the median value of
        the running averages of all completed trials’ objectives reported up to step S.

    * - :class:`Curvefitting <nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor>`
      - It stops a pending trial X at step S if
        the trial’s forecast result at target step is convergence and lower than the best performance in the history.
