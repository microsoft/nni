How to Use Tensorboard within WebUI
===================================

Users can launch a tensorboard process cross one or multi trials within webui since NNI v2.2. This feature supports local training service and reuse mode training service with shared storage for now, and will support more scenarios in later nni version.

Preparation
-----------

Make sure tensorboard installed in your environment. If you never used tensorboard, here are getting start tutorials for your reference, tensorboard with tensorflow (https://www.tensorflow.org/tensorboard/get_started), tensorboard with pytorch (https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html).

Use WebUI Launch Tensorboard
----------------------------

1. Save Logs
^^^^^^^^^^^^

NNI will automatically fetch the ``tensorboard`` subfolder under trial's output folder as tensorboard logdir. So in trial's source code, user need to save the tensorboard logs under ``NNI_OUTPUT_DIR/tensorboard``. This log path can be joined as:

.. code-block:: python

    log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')

2. Launch Tensorboard
^^^^^^^^^^^^^^^^^^^^^

3. Close All
^^^^^^^^^^^^
