Pruning Transformer with NNI
============================

Workable Pruning Process
------------------------

Here we show an effective transformer pruning process that NNI team has tried, and users can use NNI to discover better processes.
The full code can be found `here <https://github.com/microsoft/nni/tree/v2.9/examples/model_compress/pruning/pruning_bert_glue.py>`_.

In simple terms, the entire pruning process can be divided into the following steps:

1. Finetune the pretrained model on the downstream task. From our experience,
   the final effect of pruning on the finetuned model is better than pruning directly on the pretrained model.
   At the same time, the finetuned model obtained in this step will also be used as the teacher model for the following distillation training.
2. Pruning the attention layer at first. Here we apply block-sparse on attention layer weight,
   and directly prune the head (condense the weight) if the head was fully masked.
   If the head was partial masked, we will not prune it and recover its weight.
3. Retrain the head-pruned model with distillation. Recover the model precision before pruning FFN layer.
4. Pruning the FFN layer. Here we apply the output channels pruning on the 1st FFN layer,
   and the 2nd FFN layer input channels will be pruned due to the pruning of 1st layer output channels.
5. Retrain the final pruned model with distillation.

Following this process, we can get a good compression effect on transformer.

Combined with the full code, some of our recipes:

* We using :ref:`movement-pruner` in step 2 and :ref:`taylor-fo-weight-pruner` in step 4. :ref:`movement-pruner` has good performance on attention layers,
  and :ref:`taylor-fo-weight-pruner` method has good performance on FFN layers. These two pruners are all some kinds of gradient-based pruning algorithm,
  we also try weight-based pruning algorithms like :ref:`l1-norm-pruner`, but it doesn't seem to work well in this scenario.
* Distillation is a good way to recover model precision. In terms of results, usually 1~2% improvement in accuracy can be achieved when we prune bert on mnli task.
* It is necessary to gradually increase the sparsity rather than reaching a very high sparsity all at once.

Result
------

.. list-table:: Prune Bert-base-uncased on MNLI
   :header-rows: 1

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
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=1)
      - :ref:`taylor-fo-weight-pruner`
      - 51.85%
      - 84.66 / 84.88
      - -0.07 / +0.15
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=3)
      - :ref:`taylor-fo-weight-pruner`
      - 68.06%
      - 83.59 / 83.84
      - -1.14 / -0.79
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=5)
      - :ref:`taylor-fo-weight-pruner`
      - 72.69%
      - 83.19 / 83.65
      - -1.54 / -0.98
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=10)
      - :ref:`taylor-fo-weight-pruner`
      - 80.50%
      - 82.70 / 82.35
      - -2.03 / -2.28
    * - :ref:`movement-pruner` (soft, th=0.1, lambda=20)
      - :ref:`taylor-fo-weight-pruner`
      - 87.50%
      - 81.29/81.12
      - -3.44 / -3.51
