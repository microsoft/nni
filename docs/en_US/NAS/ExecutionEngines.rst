Execution Engines
=================

Pure-python execution engine (experimental)
-------------------------------------------

If you are experiencing issues with TorchScript, or the generated model code by Retiarii, there is another execution engine called Pure-python execution engine which doesn't need the code-graph conversion. This should generally not affect models and strategies in most cases, but customized mutation might not be supported.

This will come as the default execution engine in future version of Retiarii.

Three steps are needed to enable this engine now.

1. Add ``@nni.retiarii.model_wrapper`` decorator outside the whole PyTorch model.
2. Add ``config.execution_engine = 'py'`` to ``RetiariiExeConfig``.
3. If you need to export top models, formatter needs to be set to ``dict``. Exporting ``code`` won't work with this engine.

.. note:: You should always use ``super().__init__()` instead of ``super(MyNetwork, self).__init__()`` in the PyTorch model, because the latter one has issues with model wrapper.