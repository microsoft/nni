class Model(network.Network, version_utils.ModelVersionSelector):
  def __init__(self, *args, **kwargs):
    super(Model, self).__init__(*args, **kwargs)
    _keras_api_gauge.get_cell('model').set(True)
    # Model must be created under scope of DistStrat it will be trained with.
    if ds_context.has_strategy():
      self._distribution_strategy = ds_context.get_strategy()
    else:
      self._distribution_strategy = None
    # Defaults to value of `tf.config.experimental_functions_run_eagerly`.
    self._run_eagerly = None
    self.stop_training = False
    # Initialize cache attrs.
    self._reset_compile_cache()

    # Fault-tolerance handler. Set in `ModelCheckpoint`.
    self._training_state = None
    self.history = None

    # These objects are used in the default `Model.compile`. They are not
    # guaranteed to be set after `Model.compile` is called, as users can
    # override compile with custom logic.
    self.compiled_loss = None
    self.compiled_metrics = None

  def get_weights(self):
    with self.distribute_strategy.scope():
      return super(Model, self).get_weights()

  def load_weights(self, filepath, by_name=False, skip_mismatch=False):
    if dist_utils.is_tpu_strategy(self._distribution_strategy):
      if (self._distribution_strategy.extended.steps_per_run > 1 and
          (not network._is_hdf5_filepath(filepath))):  # pylint: disable=protected-access
        raise ValueError('Load weights is not yet supported with TPUStrategy '
                         'with steps_per_run greater than 1.')
    return super(Model, self).load_weights(filepath, by_name, skip_mismatch)

  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              **kwargs):
    _keras_api_gauge.get_cell('compile').set(True)
    with self.distribute_strategy.scope():
      self._validate_compile(optimizer, metrics, **kwargs)
      self._run_eagerly = kwargs.pop('run_eagerly', None)

      self.optimizer = self._get_optimizer(optimizer)
      self.compiled_loss = compile_utils.LossesContainer(
          loss, loss_weights, output_names=self.output_names)
      self.compiled_metrics = compile_utils.MetricsContainer(
          metrics, weighted_metrics, output_names=self.output_names)

      # Initializes attrs that are reset each time `compile` is called.
      self._reset_compile_cache()
      self._is_compiled = True

      self.loss = loss or {}  # Backwards compat.

  def _get_optimizer(self, optimizer):
    """Wraps `optimizer` in `LossScaleOptimizer` if necessary."""

    def _get_single_optimizer(opt):
      opt = optimizers.get(opt)
      if (self._dtype_policy.loss_scale is not None and
          not isinstance(opt, lso.LossScaleOptimizer)):
        opt = lso.LossScaleOptimizer(opt, self._dtype_policy.loss_scale)
      return opt

    return nest.map_structure(_get_single_optimizer, optimizer)

  @trackable.no_automatic_dependency_tracking
  def _reset_compile_cache(self):
    self.train_function = None
    self.test_function = None
    self.predict_function = None

    # Used to cache `trainable` attr of `Layer`s for `fit`.
    self._compiled_trainable_state = self._get_trainable_state()

  @property
  def metrics(self):
    metrics = []
    if self._is_compiled:
      # TODO(omalleyt): Track `LossesContainer` and `MetricsContainer` objects
      # so that attr names are not load-bearing.
      if self.compiled_loss is not None:
        metrics += self.compiled_loss.metrics
      if self.compiled_metrics is not None:
        metrics += self.compiled_metrics.metrics

    all_layers = self._gather_unique_layers()
    for l in all_layers:
      metrics.extend(l._metrics)  # pylint: disable=protected-access
    return metrics

  @property
  def metrics_names(self):
    return [m.name for m in self.metrics]

  @property
  def distribute_strategy(self):
    return self._distribution_strategy or ds_context.get_strategy()

  @property
  def run_eagerly(self):
    """Settable attribute indicating whether the model should run eagerly."""
    if self._run_eagerly is True and not context.executing_eagerly():
      raise ValueError('You can only set `run_eagerly=True` if eager execution '
                       'is enabled.')
    if not self.dynamic:
      if self._run_eagerly is None:
        # Respect `tf.config.experimental_run_functions_eagerly` unless
        # `run_eagerly` was explicitly passed to `compile`.
        return def_function.RUN_FUNCTIONS_EAGERLY
      else:
        return self._run_eagerly
    else:
      if not context.executing_eagerly():
        raise ValueError('Your model contains layers that can only be '
                         'successfully run in eager execution (layers '
                         'constructed with `dynamic=True`). '
                         'You must enable eager execution with '
                         '`tf.enable_eager_execution()`.')
      if self._run_eagerly is False:
        # TODO(fchollet): consider using py_func to enable this.
        raise ValueError('Your model contains layers that can only be '
                         'successfully run in eager execution (layers '
                         'constructed with `dynamic=True`). '
                         'You cannot set `run_eagerly=False`.')
      return context.executing_eagerly()

  @run_eagerly.setter
  def run_eagerly(self, value):
    self._run_eagerly = value

  def train_step(self, data):
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    with backprop.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(
          y, y_pred, sample_weight, regularization_losses=self.losses)
    # For custom training steps, users can just write:
    #   trainable_variables = self.trainable_variables
    #   gradients = tape.gradient(loss, trainable_variables)
    #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    # The _minimize call does a few extra steps unnecessary in most cases,
    # such as loss scaling and gradient clipping.
    _minimize(self.distribute_strategy, tape, self.optimizer, loss,
              self.trainable_variables)

    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return {m.name: m.result() for m in self.metrics}

  def make_train_function(self):
    if self.train_function is not None:
      return self.train_function

    def train_function(iterator):
      data = next(iterator)
      outputs = self.distribute_strategy.run(
          self.train_step, args=(data,))
      outputs = reduce_per_replica(
          outputs, self.distribute_strategy, reduction='first')
      return outputs

    if not self.run_eagerly:
      train_function = def_function.function(
          train_function, experimental_relax_shapes=True)

    self.train_function = train_function
    return self.train_function

  @enable_multi_worker
  def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          **kwargs):
    _keras_api_gauge.get_cell('fit').set(True)
    # Legacy graph support is contained in `training_v1.Model`.
    version_utils.disallow_legacy_graph('Model', 'fit')
    self._assert_compile_was_called()
    self._check_call_args('fit')

    if validation_split:
      # Create the validation data using the training data. Only supported for
      # `Tensor` and `NumPy` input.
      (x, y, sample_weight), validation_data = (
          data_adapter.train_validation_split((x, y, sample_weight),
                                              validation_split=validation_split,
                                              shuffle=False))

    with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.DataHandler(
          x=x,
          y=y,
          sample_weight=sample_weight,
          batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          initial_epoch=initial_epoch,
          epochs=epochs,
          shuffle=shuffle,
          class_weight=class_weight,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=epochs,
            steps=data_handler.inferred_steps)

      self.stop_training = False
      train_function = self.make_train_function()
      callbacks.on_train_begin()
      # Handle fault-tolerance for multi-worker.
      # TODO(omalleyt): Fix the ordering issues that mean this has to
      # happen after `callbacks.on_train_begin`.
      data_handler._initial_epoch = (  # pylint: disable=protected-access
          self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
      for epoch, iterator in data_handler.enumerate_epochs():
        self.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            with traceme.TraceMe(
                'TraceContext',
                graph_type='train',
                epoch_num=epoch,
                step_num=step,
                batch_size=batch_size):
              callbacks.on_train_batch_begin(step)
              tmp_logs = train_function(iterator)
              # Catch OutOfRangeError for Datasets of unknown size.
              # This blocks until the batch has finished executing.
              # TODO(b/150292341): Allow multiple async steps here.
              if not data_handler.inferred_steps:
                context.async_wait()
              logs = tmp_logs  # No error, now safe to assign to logs.
              callbacks.on_train_batch_end(step, logs)
        epoch_logs = copy.copy(logs)

        # Run validation.
        if validation_data and self._should_eval(epoch, validation_freq):
          val_x, val_y, val_sample_weight = (
              data_adapter.unpack_x_y_sample_weight(validation_data))
          val_logs = self.evaluate(
              x=val_x,
              y=val_y,
              sample_weight=val_sample_weight,
              batch_size=validation_batch_size or batch_size,
              steps=validation_steps,
              callbacks=callbacks,
              max_queue_size=max_queue_size,
              workers=workers,
              use_multiprocessing=use_multiprocessing,
              return_dict=True)
          val_logs = {'val_' + name: val for name, val in val_logs.items()}
          epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)
        if self.stop_training:
          break

      callbacks.on_train_end()
      return self.history

  def test_step(self, data):
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    y_pred = self(x, training=False)
    # Updates stateful loss metrics.
    self.compiled_loss(
        y, y_pred, sample_weight, regularization_losses=self.losses)

    self.compiled_metrics.update_state(y, y_pred, sample_weight)
    return {m.name: m.result() for m in self.metrics}

  def make_test_function(self):
    if self.test_function is not None:
      return self.test_function

    def test_function(iterator):
      data = next(iterator)
      outputs = self.distribute_strategy.run(
          self.test_step, args=(data,))
      outputs = reduce_per_replica(
          outputs, self.distribute_strategy, reduction='first')
      return outputs

    if not self.run_eagerly:
      test_function = def_function.function(
          test_function, experimental_relax_shapes=True)

    self.test_function = test_function
    return self.test_function

  @enable_multi_worker
  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False,
               return_dict=False):
    _keras_api_gauge.get_cell('evaluate').set(True)
    version_utils.disallow_legacy_graph('Model', 'evaluate')
    self._assert_compile_was_called()
    self._check_call_args('evaluate')

    with self.distribute_strategy.scope():
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.DataHandler(
          x=x,
          y=y,
          sample_weight=sample_weight,
          batch_size=batch_size,
          steps_per_epoch=steps,
          initial_epoch=0,
          epochs=1,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps)

      test_function = self.make_test_function()
      callbacks.on_test_begin()
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        self.reset_metrics()
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            with traceme.TraceMe(
                'TraceContext',
                graph_type='test',
                step_num=step):
              callbacks.on_test_batch_begin(step)
              tmp_logs = test_function(iterator)
              # Catch OutOfRangeError for Datasets of unknown size.
              # This blocks until the batch has finished executing.
              # TODO(b/150292341): Allow multiple async steps here.
              if not data_handler.inferred_steps:
                context.async_wait()
              logs = tmp_logs  # No error, now safe to assign to logs.
              callbacks.on_test_batch_end(step, logs)
      callbacks.on_test_end()

      logs = tf_utils.to_numpy_or_python_type(logs)
      if return_dict:
        return logs
      else:
        results = [logs.get(name, None) for name in self.metrics_names]
        if len(results) == 1:
          return results[0]
        return results

  def predict_step(self, data):
    data = data_adapter.expand_1d(data)
    x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
    return self(x, training=False)

  def make_predict_function(self):
    if self.predict_function is not None:
      return self.predict_function

    def predict_function(iterator):
      data = next(iterator)
      outputs = self.distribute_strategy.run(
          self.predict_step, args=(data,))
      outputs = reduce_per_replica(
          outputs, self.distribute_strategy, reduction='concat')
      return outputs

    if not self.run_eagerly:
      predict_function = def_function.function(
          predict_function, experimental_relax_shapes=True)

    self.predict_function = predict_function
    return self.predict_function

  @disable_multi_worker
  def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
    _keras_api_gauge.get_cell('predict').set(True)
    version_utils.disallow_legacy_graph('Model', 'predict')
    self._check_call_args('predict')

    outputs = None
    with self.distribute_strategy.scope():
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.DataHandler(
          x=x,
          batch_size=batch_size,
          steps_per_epoch=steps,
          initial_epoch=0,
          epochs=1,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=1,
            steps=data_handler.inferred_steps)

      predict_function = self.make_predict_function()
      callbacks.on_predict_begin()
      for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            callbacks.on_predict_batch_begin(step)
            tmp_batch_outputs = predict_function(iterator)
            # Catch OutOfRangeError for Datasets of unknown size.
            # This blocks until the batch has finished executing.
            # TODO(b/150292341): Allow multiple async steps here.
            if not data_handler.inferred_steps:
              context.async_wait()
            batch_outputs = tmp_batch_outputs  # No error, now safe to assign.
            if outputs is None:
              outputs = nest.map_structure(lambda batch_output: [batch_output],
                                           batch_outputs)
            else:
              nest.map_structure_up_to(
                  batch_outputs,
                  lambda output, batch_output: output.append(batch_output),
                  outputs, batch_outputs)
            callbacks.on_predict_batch_end(step, {'outputs': batch_outputs})
      callbacks.on_predict_end()
    all_outputs = nest.map_structure_up_to(batch_outputs, concat, outputs)
    return tf_utils.to_numpy_or_python_type(all_outputs)

  def reset_metrics(self):
    """Resets the state of metrics."""
    for m in self.metrics:
      m.reset_states()

  def train_on_batch(self,
                     x,
                     y=None,
                     sample_weight=None,
                     class_weight=None,
                     reset_metrics=True,
                     return_dict=False):
    self._assert_compile_was_called()
    self._check_call_args('train_on_batch')
    with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x,
                                                    y, sample_weight,
                                                    class_weight)
      train_function = self.make_train_function()
      logs = train_function(iterator)

    if reset_metrics:
      self.reset_metrics()
    logs = tf_utils.to_numpy_or_python_type(logs)
    if return_dict:
      return logs
    else:
      results = [logs.get(name, None) for name in self.metrics_names]
      if len(results) == 1:
        return results[0]
      return results

  def test_on_batch(self,
                    x,
                    y=None,
                    sample_weight=None,
                    reset_metrics=True,
                    return_dict=False):
    self._assert_compile_was_called()
    self._check_call_args('test_on_batch')
    with self.distribute_strategy.scope():
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x,
                                                    y, sample_weight)
      test_function = self.make_test_function()
      logs = test_function(iterator)

    if reset_metrics:
      self.reset_metrics()
    logs = tf_utils.to_numpy_or_python_type(logs)
    if return_dict:
      return logs
    else:
      results = [logs.get(name, None) for name in self.metrics_names]
      if len(results) == 1:
        return results[0]
      return results

  def predict_on_batch(self, x):
    self._check_call_args('predict_on_batch')
    with self.distribute_strategy.scope():
      iterator = data_adapter.single_batch_iterator(self.distribute_strategy, x)
      predict_function = self.make_predict_function()
      outputs = predict_function(iterator)
    return tf_utils.to_numpy_or_python_type(outputs)

  def _check_call_args(self, method_name):
    """Check that `call` has only one positional arg."""
    # Always allow first arg, regardless of arg name.
    fullargspec = self._call_full_argspec
    if fullargspec.defaults:
      positional_args = fullargspec.args[:-len(fullargspec.defaults)]
    else:
      positional_args = fullargspec.args
    if 'training' in positional_args:
      positional_args.remove('training')

    # self and first arg can be positional.
    if len(positional_args) > 2:
      extra_args = positional_args[2:]
      raise ValueError(
          'Models passed to `' + method_name + '` can only have `training` '
          'and the first argument in `call` as positional arguments, '
          'found: ' + str(extra_args) + '.')

  def _validate_compile(self, optimizer, metrics, **kwargs):
    """Performs validation checks for the default `compile`."""
    if any(
        isinstance(opt, optimizers.Optimizer)
        for opt in nest.flatten(optimizer)):
      raise ValueError(
          '`tf.compat.v1.keras` Optimizer (', optimizer, ') is '
          'not supported when eager execution is enabled. Use a '
          '`tf.keras` Optimizer instead, or disable eager '
          'execution.')

    kwargs.pop('cloning', None)  # Legacy DistStrat argument, never used.
    kwargs.pop('experimental_run_tf_function', None)  # Always `True`.
    if kwargs.pop('distribute', None) is not None:
      raise ValueError(
          'Distribute argument in compile is not available in TF 2.0 please '
          'create the model under the distribution strategy scope.')
    if kwargs.pop('target_tensors', None) is not None:
      raise ValueError(
          'target_tensors argument is not supported when executing eagerly.')
    invalid_kwargs = set(kwargs) - {'run_eagerly'}
    if invalid_kwargs:
      raise TypeError('Invalid keyword argument(s) in `compile`: %s' %
                      (invalid_kwargs,))

    # Model must be created and compiled with the same DistStrat.
    if self.built and ds_context.has_strategy():
      strategy = ds_context.get_strategy()
      for v in self.variables:
        if not strategy.extended.variable_created_in_scope(v):
          raise ValueError(
              'Variable (%s) was not created in the distribution strategy '
              'scope of (%s). It is most likely due to not all layers or '
              'the model or optimizer being created outside the distribution '
              'strategy scope. Try to make sure your code looks similar '
              'to the following.\n'
              'with strategy.scope():\n'
              '  model=_create_model()\n'
              '  model.compile(...)' % (v, strategy))

    # Model metrics must be created in the same distribution strategy scope
    # as the model.
    strategy = self._get_distribution_strategy()
    for metric in nest.flatten(metrics):
      for v in getattr(metric, 'variables', []):
        if not strategy.extended.variable_created_in_scope(v):
          raise ValueError(
              'Metric (%s) passed to model.compile was created inside of a '
              'different distribution strategy scope than the model. All '
              'metrics must be created in the same distribution strategy '
              'scope as the model (in this case %s). If you pass in a string '
              'identifier for a metric to compile the metric will '
              'automatically be created in the correct distribution '
              'strategy scope.' % (metric, strategy)
          )

  def _maybe_load_initial_epoch_from_ckpt(self, initial_epoch):
    """Maybe load initial epoch from ckpt considering possible worker recovery.

    Refer to tensorflow/python/keras/distribute/multi_worker_training_state.py
    for more information.

    Arguments:
      initial_epoch: The original initial_epoch user passes in in `fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """
    if self._training_state is not None:
      return self._training_state.maybe_load_initial_epoch_from_ckpt(
          initial_epoch, mode=ModeKeys.TRAIN)
    return initial_epoch

  def _assert_compile_was_called(self):
    # Checks whether `compile` has been called. If it has been called,
    # then the optimizer is set. This is different from whether the
    # model is compiled
    # (i.e. whether the model is built and its inputs/outputs are set).
    if not self._is_compiled:
      raise RuntimeError('You must compile your model before '
                         'training/testing. '
                         'Use `model.compile(optimizer, loss)`.')

  def _set_inputs(self, inputs, outputs=None, training=None):
    """This method is for compat with Modelv1. Only inputs are needed here."""
    self._set_save_spec(inputs)

  @property
  def _trackable_saved_model_saver(self):
    return model_serialization.ModelSavedModelSaver(self)

  def _list_functions_for_serialization(self, serialization_cache):
    # SavedModel needs to ignore the execution functions.
    train_function = self.train_function
    test_function = self.test_function
    predict_function = self.predict_function
    self.train_function = None
    self.test_function = None
    self.predict_function = None
    functions = super(
        Model, self)._list_functions_for_serialization(serialization_cache)
    self.train_function = train_function
    self.test_function = test_function
    self.predict_function = predict_function
    return functions

  def _should_eval(self, epoch, validation_freq):
    epoch = epoch + 1  # one-index the user-facing epoch.
    if isinstance(validation_freq, int):
      return epoch % validation_freq == 0
    elif isinstance(validation_freq, list):
      return epoch in validation_freq
    else:
      raise ValueError('Expected `validation_freq` to be a list or int.')
