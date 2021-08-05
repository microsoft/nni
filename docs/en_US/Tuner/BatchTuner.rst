Batch Tuner on NNI
==================

Batch Tuner
-----------

Batch tuner allows users to simply provide several configurations (i.e., choices of hyper-parameters) for their trial code. After finishing all the configurations, the experiment is done. Batch tuner only supports the type ``choice`` in the `search space spec <../Tutorial/SearchSpaceSpec.rst>`__.

Suggested scenario: If the configurations you want to try have been decided, you can list them in the SearchSpace file (using ``choice``\ ) and run them using the batch tuner.
