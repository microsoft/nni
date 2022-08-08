Pruning Transformer with NNI
============================

Workable Pruning Process
------------------------

Here we show an effective transformer pruning process that NNI team has tried, and users can use NNI to discover better processes.
The full code can be found :githublink:`here <examples/model_compress/pruning/pruning_bert_glue.py>`.

During the process of pruning transformer, we gained some of the following experience:

* We using :ref:`movement-pruner` in step 2 and :ref:`taylor-fo-weight-pruner` in step 4. :ref:`movement-pruner` has good performance on attention layers,
  and :ref:`taylor-fo-weight-pruner` method has good performance on FFN layers. These two pruners are all some kinds of gradient-based pruning algorithm,
  we also try weight-based pruning algorithms like :ref:`l1-norm-pruner`, but it doesn't seem to work well in this scenario.
* Distillation is a good way to recover model precision. In terms of results, usually 1~2% improvement in accuracy can be achieved when we prune bert on mnli task.
* It is necessary to gradually increase the sparsity rather than reaching a very high sparsity all at once.

Result
------

.. list-table:: Prune Bert-base-uncased on MNLI
    :header-rows: 1
    :widths: auto

    * - Attention Pruning Method
      - FFN Pruning Method
      - Total Sparsity
      - Accuracy
      - Acc. Drop
    * -
      -
      - 0%
      - 84.73 / 84.63
      - +0.0 / +0.0
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=5)
      - :ref:`taylor-fo-weight-pruner`
      - 51.39%
      - 84.25 / 84.96
      - -0.48 / +0.33
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=10)
      - :ref:`taylor-fo-weight-pruner`
      - 66.67%
      - 83.98 / 83.75
      - -0.75 / -0.88
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=20)
      - :ref:`taylor-fo-weight-pruner`
      - 77.78%
      - 83.02 / 83.06
      - -1.71 / -1.57
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=30)
      - :ref:`taylor-fo-weight-pruner`
      - 89.81%
      - 81.24 / 80.99
      - -3.49 / -3.64
