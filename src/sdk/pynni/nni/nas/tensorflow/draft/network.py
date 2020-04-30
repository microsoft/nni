class Network(base_layer.Layer):
  # See tf.Module for the usage of this property.
  # The key of _layer_call_argspecs is a layer. tf.Module._flatten will fail to
  # flatten the key since it is trying to convert Trackable/Layer to a string.
  _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
      ('_layer_call_argspecs', '_compiled_trainable_state',
       '_output_mask_cache', '_output_tensor_cache', '_output_shape_cache'),
      base_layer.Layer._TF_MODULE_IGNORED_PROPERTIES
  ))

  def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
    # Signature detection
    if (len(args) == 2 or
        len(args) == 1 and 'outputs' in kwargs or
        'inputs' in kwargs and 'outputs' in kwargs):
      # Graph network
      self._init_graph_network(*args, **kwargs)
    else:
      # Subclassed network
      self._init_subclassed_network(**kwargs)

    tf_utils.assert_no_legacy_layers(self.layers)

  # Several Network methods have "no_automatic_dependency_tracking"
  # annotations. Since Network does automatic dependency tracking on attribute
  # assignment, including for common data structures such as lists, by default
  # we'd have quite a few empty dependencies which users don't care about (or
  # would need some way to ignore dependencies automatically, which is confusing
  # when applied to user code). Some attributes, such as _layers, would cause
  # structural issues (_layers being the place where Layers assigned to tracked
  # attributes are stored).
  #
  # Aside from these aesthetic and structural issues, useless dependencies on
  # empty lists shouldn't cause issues; adding or removing them will not break
  # checkpoints, but may cause "all Python objects matched" assertions to fail
  # (in which case less strict assertions may be substituted if necessary).
  @trackable.no_automatic_dependency_tracking
  def _base_init(self, name=None, **kwargs):
    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # self.input_spec
    # self.losses
    # self.updates

    generic_utils.validate_kwargs(kwargs, {'trainable', 'dtype', 'dynamic',
                                           'autocast'})

    super(Network, self).__init__(name=name, **kwargs)

    self.output_names = None
    self.input_names = None
    self._is_compiled = False
    self._saved_model_inputs_spec = None

    # This is True for Sequential networks and Functional networks.
    self._compute_output_and_mask_jointly = False

    if not hasattr(self, 'optimizer'):
      # Don't reset optimizer if already set.
      self.optimizer = None

    self._scope = None  # Never used.
    self._reuse = None  # Never used.
    if context.executing_eagerly():
      self._graph = None
    else:
      self._graph = ops.get_default_graph()  # Used in symbolic mode only.

    self._trackable_saver = (
        trackable_utils.saver_with_op_caching(self))

  @trackable.no_automatic_dependency_tracking
  def _init_graph_network(self, inputs, outputs, name=None, **kwargs):
    generic_utils.validate_kwargs(
        kwargs, {'trainable'},
        'Functional models may only specify `name` and `trainable` keyword '
        'arguments during initialization. Got an unexpected argument:')
    # Normalize and set self.inputs, self.outputs.
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
      inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
      outputs = outputs[0]
    self._nested_outputs = outputs
    self._nested_inputs = inputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)

    # Models constructed with a single Tensor or list of Tensors can
    # be called with a dict, where the keys of the dict are the names
    # of the `Input` objects. Extra keys are ignored.
    self._enable_dict_to_input_mapping = (
        not nest.is_sequence(self._nested_inputs) or
        (isinstance(self._nested_inputs, (list, tuple)) and
         not any(nest.is_sequence(t) for t in self._nested_inputs)))

    if any(not hasattr(tensor, '_keras_history') for tensor in self.outputs):
      base_layer_utils.create_keras_history(self._nested_outputs)

    self._base_init(name=name, **kwargs)
    self._validate_graph_inputs_and_outputs()

    # A Network does not create weights of its own, thus it is already
    # built.
    self.built = True
    self._build_input_shape = nest.map_structure(lambda x: x.shape, inputs)
    self._compute_output_and_mask_jointly = True
    self._is_graph_network = True
    # `_expects_training_arg` is True since the `training` argument is always
    # present in the signature of the `call` method of a graph network.
    self._expects_training_arg = True
    self._expects_mask_arg = True
    # A graph network does not autocast inputs, as its layers will cast them
    # instead.
    self._autocast = False

    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []

    self._supports_ragged_inputs = None

    # This is for performance optimization when calling the Network on new
    # inputs. Every time the Network is called on a set on input tensors,
    # we compute the output tensors, output masks and output shapes in one pass,
    # then cache them here. When any of these outputs is queried later, we
    # retrieve it from there instead of recomputing it.
    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}

    # Build self._output_layers:
    for x in self.outputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      self._output_layers.append(layer)
      self._output_coordinates.append((layer, node_index, tensor_index))

    # Build self._input_layers:
    for x in self.inputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      # It's supposed to be an input layer, so only one node
      # and one tensor output.
      assert node_index == 0
      assert tensor_index == 0
      self._input_layers.append(layer)
      self._input_coordinates.append((layer, node_index, tensor_index))

    # Keep track of the network's nodes and layers.
    nodes, nodes_by_depth, layers, _ = _map_graph_network(
        self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._layers = layers
    self._layer_call_argspecs = {}
    for layer in self._layers:
      self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
      layer._attribute_sentinel.add_parent(self._attribute_sentinel)

    # Create the node linking internal inputs to internal outputs.
    node_module.Node(
        outbound_layer=self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=self._nested_inputs,
        output_tensors=self._nested_outputs)

    # Build self.input_names and self.output_names.
    self._set_output_names()
    self.input_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for layer in self._input_layers:
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        # Use batch_input_shape here because non-eager composite tensors may not
        # have a shape attribute that's meaningful (sparse, for instance, has
        # a tensor that's non-constant and needs to be fed). This means that
        # input layers that create placeholders will need to have the
        # batch_input_shape attr to allow for input shape validation.
        self._feed_input_shapes.append(layer._batch_input_shape)
        self._feed_inputs.append(layer.input)

    self._compute_tensor_usage_count()
    self._set_save_spec(self._nested_inputs)

  def _set_output_names(self):
    """Assigns unique names to the Network's outputs.

    Output layers with multiple output tensors would otherwise lead to duplicate
    names in self.output_names.
    """
    uniquified = []
    output_names = set()
    prefix_count = {}
    for layer in self._output_layers:
      proposal = layer.name
      while proposal in output_names:
        existing_count = prefix_count.get(layer.name, 1)
        proposal = '{}_{}'.format(layer.name, existing_count)
        prefix_count[layer.name] = existing_count + 1
      output_names.add(proposal)
      uniquified.append(proposal)
    self.output_names = uniquified

  @trackable.no_automatic_dependency_tracking
  def _init_subclassed_network(self, name=None, **kwargs):
    self._base_init(name=name, **kwargs)
    self._is_graph_network = False
    self._init_call_fn_args()
    self._autocast = kwargs.get('autocast',
                                base_layer_utils.v2_dtype_behavior_enabled())
    self._supports_ragged_inputs = None
    self.outputs = None
    self.inputs = None
    self.built = False
    self._build_input_shape = None

  @property
  @trackable_layer_utils.cache_recursive_attribute('dynamic')
  def dynamic(self):
    if self._is_graph_network:
      return any(layer.dynamic for layer in self.layers)
    return self._dynamic or any(layer.dynamic for layer in self.layers)

  @property
  def _layer_checkpoint_dependencies(self):
    """Dictionary of layer dependencies to be included in the checkpoint."""
    # Use getattr because this function can be called from __setattr__, at which
    # point the _is_graph_network attribute has not been created.
    if (not getattr(self, '_is_graph_network', False) and
        base_layer_utils.is_subclassed(self)):
      return {}  # Only add layer dependencies for graph networks

    weight_layer_index = 0

    dependencies = {}
    for layer_index, layer in enumerate(self.layers):
      try:
        if layer.weights:
          # Keep a separate index for layers which have weights. This allows
          # users to insert Layers without weights anywhere in the network
          # without breaking checkpoints.
          dependencies['layer_with_weights-%d' % weight_layer_index] = layer
          weight_layer_index += 1
      except ValueError:
        # The layer might have weights, but may not be built yet. We just treat
        # it as layer without weight.
        pass

      # Even if it doesn't have weights, we should still track everything in
      # case it has/will have Trackable dependencies.
      dependencies['layer-%d' % layer_index] = layer
    return dependencies

  @property
  def _checkpoint_dependencies(self):
    dependencies = [
        trackable.TrackableReference(name=name, ref=layer)
        for name, layer in self._layer_checkpoint_dependencies.items()]
    dependencies.extend(super(Network, self)._checkpoint_dependencies)
    return dependencies

  def _lookup_dependency(self, name):
    layer_dependencies = self._layer_checkpoint_dependencies
    if name in layer_dependencies:
      return layer_dependencies[name]
    return super(Network, self)._lookup_dependency(name)

  def _handle_deferred_layer_dependencies(self, layers):
    """Handles layer checkpoint dependencies that are added after init."""
    layer_checkpoint_dependencies = self._layer_checkpoint_dependencies
    layer_to_name = {v: k for k, v in layer_checkpoint_dependencies.items()}
    for layer in layers:
      if layer in layer_to_name:
        self._handle_deferred_dependencies(name=layer_to_name[layer],
                                           trackable=layer)

  def __setattr__(self, name, value):
    if not getattr(self, '_self_setattr_tracking', True):
      super(Network, self).__setattr__(name, value)
      return

    if all(
        isinstance(v, (base_layer.Layer,
                       data_structures.TrackableDataStructure)) or
        trackable_layer_utils.has_weights(v) for v in nest.flatten(value)):
      try:
        self._is_graph_network
      except AttributeError:
        # six.raise_from supresses the original AttributeError from being raised
        six.raise_from(
            RuntimeError('It looks like you are subclassing `Model` and you '
                         'forgot to call `super(YourClass, self).__init__()`.'
                         ' Always start with this line.'), None)

    super(Network, self).__setattr__(name, value)

    # Keep track of metric instance created in subclassed model/layer.
    # We do this so that we can maintain the correct order of metrics by adding
    # the instance to the `metrics` list as soon as it is created.
    from tensorflow.python.keras import metrics as metrics_module  # pylint: disable=g-import-not-at-top
    if isinstance(value, metrics_module.Metric):
      self._metrics.append(value)

  @property
  @trackable_layer_utils.cache_recursive_attribute('stateful')
  def stateful(self):
    return any(getattr(layer, 'stateful', False) for layer in self.layers)

  def reset_states(self):
    for layer in self.layers:
      if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
        layer.reset_states()

  @property
  def state_updates(self):
    """Returns the `updates` from all layers that are stateful.

    This is useful for separating training updates and
    state updates, e.g. when we need to update a layer's internal state
    during prediction.

    Returns:
        A list of update ops.
    """
    state_updates = []
    for layer in self.layers:
      if getattr(layer, 'stateful', False):
        if hasattr(layer, 'updates'):
          state_updates += layer.updates
    return state_updates

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self._dedup_weights(self._undeduplicated_weights)

  @property
  def _undeduplicated_weights(self):
    """Returns the undeduplicated list of all layer variables/weights."""
    self._assert_weights_created()
    weights = []
    for layer in self._layers:
      weights += layer.weights
    weights += (self._trainable_weights + self._non_trainable_weights)
    return weights

  @property
  @tracking.cached_per_instance
  def _should_compute_mask(self):
    return self._is_graph_network and super(Network, self)._should_compute_mask

  def compute_mask(self, inputs, mask):
    if not self._is_graph_network:
      return None

    # TODO(omalleyt): b/123540974 This function is not really safe to call
    # by itself because it will duplicate any updates and losses in graph
    # mode by `call`ing the Layers again.
    output_tensors = self._run_internal_graph(inputs, mask=mask)
    return nest.map_structure(lambda t: t._keras_mask, output_tensors)

  @property
  def layers(self):
    return list(
        trackable_layer_utils.filter_empty_layer_containers(self._layers))

  def get_layer(self, name=None, index=None):
    """Retrieves a layer based on either its name (unique) or index.

    If `name` and `index` are both provided, `index` will take precedence.
    Indices are based on order of horizontal graph traversal (bottom-up).

    Arguments:
        name: String, name of layer.
        index: Integer, index of layer.

    Returns:
        A layer instance.

    Raises:
        ValueError: In case of invalid layer name or index.
    """
    # TODO(fchollet): We could build a dictionary based on layer names
    # since they are constant, but we have not done that yet.
    if index is not None:
      if len(self.layers) <= index:
        raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                         ' but model only has ' + str(len(self.layers)) +
                         ' layers.')
      else:
        return self.layers[index]
    else:
      if not name:
        raise ValueError('Provide either a layer name or layer index.')
    for layer in self.layers:
      if layer.name == name:
        return layer
    raise ValueError('No such layer: ' + name)

  @property
  def trainable_weights(self):
    self._assert_weights_created()
    return self._dedup_weights(
        trackable_layer_utils.gather_trainable_weights(
            trainable=self.trainable,
            sub_layers=self._layers,
            extra_variables=self._trainable_weights))

  @property
  def non_trainable_weights(self):
    self._assert_weights_created()
    return self._dedup_weights(
        trackable_layer_utils.gather_non_trainable_weights(
            trainable=self.trainable,
            sub_layers=self._layers,
            extra_variables=self._non_trainable_weights +
            self._trainable_weights))

  @property
  def input_spec(self):
    """Gets the network's input specs.

    Returns:
        A list of `InputSpec` instances (one per input to the model)
            or a single instance if the model has only one input.
    """
    return

  @generic_utils.default
  def build(self, input_shape):
    """Builds the model based on input shapes received.

    This is to be used for subclassed models, which do not know at instantiation
    time what their inputs look like.

    This method only exists for users who want to call `model.build()` in a
    standalone way (as a substitute for calling the model on real data to
    build it). It will never be called by the framework (and thus it will
    never throw unexpected errors in an unrelated workflow).

    Args:
     input_shape: Single tuple, TensorShape, or list of shapes, where shapes
         are tuples, integers, or TensorShapes.

    Raises:
      ValueError:
        1. In case of invalid user-provided data (not of type tuple,
           list, or TensorShape).
        2. If the model requires call arguments that are agnostic
           to the input shapes (positional or kwarg in call signature).
        3. If not all layers were properly built.
        4. If float type inputs are not supported within the layers.

      In each of these cases, the user should build their model by calling it
      on real tensor data.
    """
    if self._is_graph_network:
      super(Network, self).build(input_shape)
      return

    # If subclass network
    if input_shape is None:
      raise ValueError('Input shape must be defined when calling build on a '
                       'model subclass network.')
    valid_types = (tuple, list, tensor_shape.TensorShape)
    if not isinstance(input_shape, valid_types):
      raise ValueError('Specified input shape is not one of the valid types. '
                       'Please specify a batch input shape of type tuple or '
                       'list of input shapes. User provided '
                       'input type: {}'.format(type(input_shape)))

    if input_shape and not self.inputs:
      # We create placeholders for the `None`s in the shape and build the model
      # in a Graph. Since tf.Variable is compatible with both eager execution
      # and graph building, the variables created after building the model in
      # a Graph are still valid when executing eagerly.
      if context.executing_eagerly():
        graph = func_graph.FuncGraph('build_graph')
      else:
        graph = backend.get_graph()
      with graph.as_default():
        if isinstance(input_shape, list):
          x = [base_layer_utils.generate_placeholders_from_shape(shape)
               for shape in input_shape]
        elif isinstance(input_shape, dict):
          x = {
              k: base_layer_utils.generate_placeholders_from_shape(shape)
              for k, shape in input_shape.items()
          }
        else:
          x = base_layer_utils.generate_placeholders_from_shape(input_shape)

        kwargs = {}
        call_signature = self._call_full_argspec
        call_args = call_signature.args
        # Exclude `self`, `inputs`, and any argument with a default value.
        if len(call_args) > 2:
          if call_signature.defaults:
            call_args = call_args[2:-len(call_signature.defaults)]
          else:
            call_args = call_args[2:]
          for arg in call_args:
            if arg == 'training':
              # Case where `training` is a positional arg with no default.
              kwargs['training'] = False
            else:
              # Has invalid call signature with unknown positional arguments.
              raise ValueError(
                  'Currently, you cannot build your model if it has '
                  'positional or keyword arguments that are not '
                  'inputs to the model, but are required for its '
                  '`call` method. Instead, in order to instantiate '
                  'and build your model, `call` your model on real '
                  'tensor data with all expected call arguments.')
        elif len(call_args) < 2:
          # Signature without `inputs`.
          raise ValueError('You can only call `build` on a model if its `call` '
                           'method accepts an `inputs` argument.')
        try:
          self.call(x, **kwargs)
        except (errors.InvalidArgumentError, TypeError):
          raise ValueError('You cannot build your model by calling `build` '
                           'if your layers do not support float type inputs. '
                           'Instead, in order to instantiate and build your '
                           'model, `call` your model on real tensor data (of '
                           'the correct dtype).')

    super(Network, self).build(input_shape)

  def call(self, inputs, training=None, mask=None):
    if not self._is_graph_network:
      raise NotImplementedError('When subclassing the `Model` class, you should'
                                ' implement a `call` method.')

    return self._run_internal_graph(
        inputs, training=training, mask=mask,
        convert_kwargs_to_constants=base_layer_utils.call_context().saving)

  def compute_output_shape(self, input_shape):
    if not self._is_graph_network:
      return super(Network, self).compute_output_shape(input_shape)

    # Convert any shapes in tuple format to TensorShapes.
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

    if len(nest.flatten(input_shape)) != len(nest.flatten(self._input_layers)):
      raise ValueError('Invalid input_shape argument ' + str(input_shape) +
                       ': model has ' + str(len(self._input_layers)) +
                       ' tensor inputs.')

    # Use the tuple of TensorShape as the cache key, since tuple is hashable
    # and can be used as hash key.
    try:
      cache_key = tuple(tf_utils.convert_shapes(input_shape, to_tuples=True))
      if cache_key in self._output_shape_cache:
        # Cache hit. Return shapes as TensorShapes.
        return self._output_shape_cache[cache_key]
    except ValueError:
      # In case there are unknown TensorShape, eg for sparse tensor input,
      # We skip the caching since the shape is unknown.
      pass

    layers_to_output_shapes = {}
    for layer, shape in zip(self._input_layers, nest.flatten(input_shape)):
      # It's an input layer: then `compute_output_shape` is identity,
      # and there is only one node and one tensor..
      shape_key = layer.name + '_0_0'
      layers_to_output_shapes[shape_key] = shape

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    # Iterate over nodes, by depth level.
    if len(depth_keys) > 1:
      for depth in depth_keys:
        nodes = self._nodes_by_depth[depth]
        for node in nodes:
          # This is always a single layer, never a list.
          layer = node.outbound_layer
          if layer in self._input_layers:
            # We've already covered the input layers
            # a few lines above.
            continue
          # Potentially redundant list,
          # same size as node.input_tensors.
          layer_input_shapes = []
          for inbound_layer, node_id, tensor_id, _ in node.iterate_inbound():
            input_layer_key = inbound_layer.name + '_%s_%s' % (node_id,
                                                               tensor_id)
            layer_input_shapes.append(layers_to_output_shapes[input_layer_key])
          layer_input_shapes = nest.pack_sequence_as(node.inbound_layers,
                                                     layer_input_shapes)
          # Layers expect shapes to be tuples for `compute_output_shape`.
          layer_input_shapes = tf_utils.convert_shapes(
              layer_input_shapes, to_tuples=True)
          layer_output_shapes = layer.compute_output_shape(layer_input_shapes)
          # Convert back to TensorShapes.
          layer_output_shapes = tf_utils.convert_shapes(
              layer_output_shapes, to_tuples=False)

          node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access
          for j, shape in enumerate(nest.flatten(layer_output_shapes)):
            shape_key = layer.name + '_%s_%s' % (node_index, j)
            layers_to_output_shapes[shape_key] = shape

      # Read final output shapes from layers_to_output_shapes.
      output_shapes = []
      for i in range(len(self._output_layers)):
        layer, node_index, tensor_index = self._output_coordinates[i]
        shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
        output_shapes.append(layers_to_output_shapes[shape_key])
      output_shapes = nest.pack_sequence_as(self._nested_outputs, output_shapes)
      # Store in cache.
      self._output_shape_cache[cache_key] = output_shapes

    # Return shapes as TensorShapes.
    return output_shapes

  def _run_internal_graph(self, inputs, training=None, mask=None,
                          convert_kwargs_to_constants=False):
    inputs = self._flatten_to_reference_inputs(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = self._flatten_to_reference_inputs(mask)
    for input_t, mask in zip(inputs, masks):
      input_t._keras_mask = mask

    # Dictionary mapping reference tensors to computed tensors.
    tensor_dict = {}
    for x, y in zip(self.inputs, inputs):
      y = self._conform_to_reference_input(y, ref_input=x)
      x_id = str(id(x))
      tensor_dict[x_id] = [y] * self._tensor_usage_count[x_id]

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    # Ignore the InputLayers when computing the graph.
    depth_keys = depth_keys[1:]

    for depth in depth_keys:
      nodes = self._nodes_by_depth[depth]
      for node in nodes:
        # This is always a single layer, never a list.
        layer = node.outbound_layer

        if all(
            str(id(tensor)) in tensor_dict
            for tensor in nest.flatten(node.input_tensors)):

          # Call layer (reapplying ops to new inputs).
          computed_tensors = nest.map_structure(
              lambda t: tensor_dict[str(id(t))].pop(), node.input_tensors)

          # Ensure `training` arg propagation if applicable.
          kwargs = copy.copy(node.arguments) if node.arguments else {}
          if convert_kwargs_to_constants:
            kwargs = _map_tensors_to_constants(kwargs)

          argspec = self._layer_call_argspecs[layer].args
          if 'training' in argspec:
            if 'training' not in kwargs or kwargs['training'] is None:
              kwargs['training'] = training
            elif (type(kwargs['training']) is ops.Tensor and  # pylint: disable=unidiomatic-typecheck
                  any([
                      kwargs['training'] is x
                      for x in backend._GRAPH_LEARNING_PHASES.values()
                  ])):
              kwargs['training'] = training  # Materialize placeholder.

          # Map Keras tensors in kwargs to their computed value.
          def _map_tensor_if_from_keras_layer(t):
            if (isinstance(t,
                           (ops.Tensor, composite_tensor.CompositeTensor)) and
                hasattr(t, '_keras_history')):
              t_id = str(id(t))
              return tensor_dict[t_id].pop()
            return t

          kwargs = nest.map_structure(_map_tensor_if_from_keras_layer, kwargs)

          # Compute outputs.
          output_tensors = layer(computed_tensors, **kwargs)

          # Update tensor_dict.
          for x, y in zip(
              nest.flatten(node.output_tensors), nest.flatten(output_tensors)):
            x_id = str(id(x))
            tensor_dict[x_id] = [y] * self._tensor_usage_count[x_id]

    output_tensors = []
    output_shapes = []
    for x in self.outputs:
      assert str(id(x)) in tensor_dict, 'Could not compute output ' + str(x)
      tensor = tensor_dict[str(id(x))].pop()
      output_shapes.append(x.shape)
      output_tensors.append(tensor)

    if output_shapes is not None:
      input_shapes = [x.shape for x in inputs]
      try:
        cache_key = tuple(tf_utils.convert_shapes(input_shapes, to_tuples=True))
        self._output_shape_cache[cache_key] = nest.pack_sequence_as(
            self._nested_outputs, output_shapes)
      except ValueError:
        # In case there are unknown TensorShape, eg for sparse tensor input,
        # We skip the caching since the shape is unknown.
        pass

    output_tensors = nest.pack_sequence_as(self._nested_outputs, output_tensors)
    return output_tensors

  def _flatten_to_reference_inputs(self, tensors):
    """Maps `tensors` to their respective `keras.Input`."""
    if self._enable_dict_to_input_mapping and isinstance(tensors, dict):
      ref_inputs = self._nested_inputs
      if not nest.is_sequence(ref_inputs):
        ref_inputs = [self._nested_inputs]

      # Flatten in the order the `Input`s were passed during Model construction.
      return [tensors[inp._keras_history.layer.name] for inp in ref_inputs]

    # Otherwise both self.inputs and tensors will already be in same order.
    return nest.flatten(tensors)

  def _conform_to_reference_input(self, tensor, ref_input):
    """Set shape and dtype based on `keras.Input`s."""
    # Shape handling (only for non-CompositeTensors).
    if isinstance(tensor, ops.Tensor) and isinstance(ref_input, ops.Tensor):
      # Allow (None,) and (None, 1) Tensors to be passed interchangably. Use the
      # shape specified by the `keras.Input`.
      if tensor.shape.rank is not None and ref_input.shape.rank is not None:
        should_squeeze_last_dim = (
            tensor.shape.rank == ref_input.shape.rank + 1 and
            tensor.shape[-1] == 1)
        should_expand_last_dim = (
            tensor.shape.rank == ref_input.shape.rank - 1 and
            ref_input.shape[-1] == 1)
        if should_squeeze_last_dim:
          tensor = array_ops.squeeze_v2(tensor, axis=-1)
        elif should_expand_last_dim:
          tensor = array_ops.expand_dims_v2(tensor, axis=-1)

      # Add shape hints to Tensors that might have None shape dims but have
      # shapes defined by the `keras.Input`.
      try:
        tensor.set_shape(tensor.shape.merge_with(ref_input.shape))
      except ValueError:
        logging.warning(
            'Model was constructed with shape {} for input {}, but it was '
            'called on an input with incompatible shape {}.'.format(
                ref_input.shape, ref_input, tensor.shape))

    # Dtype handling.
    if isinstance(ref_input, (ops.Tensor, composite_tensor.CompositeTensor)):
      tensor = math_ops.cast(tensor, dtype=ref_input.dtype)

    return tensor

  def get_config(self):
    if not self._is_graph_network:
      raise NotImplementedError
    return copy.deepcopy(get_network_config(self))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    input_tensors, output_tensors, created_layers = reconstruct_from_config(
        config, custom_objects)
    model = cls(inputs=input_tensors, outputs=output_tensors,
                name=config.get('name'))
    connect_ancillary_layers(model, created_layers)
    return model

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None):
    save.save_model(self, filepath, overwrite, include_optimizer, save_format,
                    signatures, options)

  def save_weights(self, filepath, overwrite=True, save_format=None):
    self._assert_weights_created()
    filepath_is_h5 = _is_hdf5_filepath(filepath)
    if save_format is None:
      if filepath_is_h5:
        save_format = 'h5'
      else:
        save_format = 'tf'
    else:
      user_format = save_format.lower().strip()
      if user_format in ('tensorflow', 'tf'):
        save_format = 'tf'
      elif user_format in ('hdf5', 'h5', 'keras'):
        save_format = 'h5'
      else:
        raise ValueError(
            'Unknown format "%s". Was expecting one of {"tf", "h5"}.' % (
                save_format,))
    if save_format == 'tf' and filepath_is_h5:
      raise ValueError(
          ('save_weights got save_format="tf"/"tensorflow", but the '
           'filepath ("%s") looks like an HDF5 file. Omit the ".h5"/".keras" '
           'when saving in TensorFlow format.')
          % filepath)

    if save_format == 'h5' and h5py is None:
      raise ImportError(
          '`save_weights` requires h5py when saving in hdf5.')
    if save_format == 'tf':
      check_filepath = filepath + '.index'
    else:
      check_filepath = filepath
    # If file exists and should not be overwritten:
    if not overwrite and os.path.isfile(check_filepath):
      proceed = ask_to_proceed_with_overwrite(check_filepath)
      if not proceed:
        return
    if save_format == 'h5':
      with h5py.File(filepath, 'w') as f:
        hdf5_format.save_weights_to_hdf5_group(f, self.layers)
    else:
      if context.executing_eagerly():
        session = None
      else:
        session = backend.get_session()
      optimizer = getattr(self, 'optimizer', None)
      if (optimizer
          and not isinstance(optimizer, trackable.Trackable)):
        logging.warning(
            ('This model was compiled with a Keras optimizer (%s) but is being '
             'saved in TensorFlow format with `save_weights`. The model\'s '
             'weights will be saved, but unlike with TensorFlow optimizers in '
             'the TensorFlow format the optimizer\'s state will not be '
             'saved.\n\nConsider using a TensorFlow optimizer from `tf.train`.')
            % (optimizer,))
      self._trackable_saver.save(filepath, session=session)
      # Record this checkpoint so it's visible from tf.train.latest_checkpoint.
      checkpoint_management.update_checkpoint_state_internal(
          save_dir=os.path.dirname(filepath),
          model_checkpoint_path=filepath,
          save_relative_paths=True,
          all_model_checkpoint_paths=[filepath])

  def load_weights(self, filepath, by_name=False, skip_mismatch=False):
    if skip_mismatch and not by_name:
      raise ValueError(
          'When calling model.load_weights, skip_mismatch can only be set to '
          'True when by_name is True.')

    if _is_hdf5_filepath(filepath):
      save_format = 'h5'
    else:
      try:
        py_checkpoint_reader.NewCheckpointReader(filepath)
        save_format = 'tf'
      except errors_impl.DataLossError:
        # The checkpoint is not readable in TensorFlow format. Try HDF5.
        save_format = 'h5'
    if save_format == 'tf':
      status = self._trackable_saver.restore(filepath)
      if by_name:
        raise NotImplementedError(
            'Weights may only be loaded based on topology into Models when '
            'loading TensorFlow-formatted weights (got by_name=True to '
            'load_weights).')
      if not context.executing_eagerly():
        session = backend.get_session()
        # Restore existing variables (if any) immediately, and set up a
        # streaming restore for any variables created in the future.
        trackable_utils.streaming_restore(status=status, session=session)
      status.assert_nontrivial_match()
      return status
    if h5py is None:
      raise ImportError(
          '`load_weights` requires h5py when loading weights from HDF5.')
    if self._is_graph_network and not self.built:
      raise NotImplementedError(
          'Unable to load weights saved in HDF5 format into a subclassed '
          'Model which has not created its variables yet. Call the Model '
          'first, then load the weights.')
    self._assert_weights_created()
    with h5py.File(filepath, 'r') as f:
      if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
      if by_name:
        hdf5_format.load_weights_from_hdf5_group_by_name(
            f, self.layers, skip_mismatch=skip_mismatch)
      else:
        hdf5_format.load_weights_from_hdf5_group(f, self.layers)

  def _updated_config(self):
    """Util shared between different serialization methods.

    Returns:
        Model config with Keras version information added.
    """
    from tensorflow.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

    config = self.get_config()
    model_config = {
        'class_name': self.__class__.__name__,
        'config': config,
        'keras_version': keras_version,
        'backend': backend.backend()
    }
    return model_config

  def to_json(self, **kwargs):
    model_config = self._updated_config()
    return json.dumps(
        model_config, default=serialization.get_json_type, **kwargs)

  def to_yaml(self, **kwargs):
    if yaml is None:
      raise ImportError(
          'Requires yaml module installed (`pip install pyyaml`).')
    return yaml.dump(self._updated_config(), **kwargs)

  def summary(self, line_length=None, positions=None, print_fn=None):
    if not self.built:
      raise ValueError('This model has not yet been built. '
                       'Build the model first by calling `build()` or calling '
                       '`fit()` with some data, or specify '
                       'an `input_shape` argument in the first layer(s) for '
                       'automatic build.')
    layer_utils.print_summary(self,
                              line_length=line_length,
                              positions=positions,
                              print_fn=print_fn)

  def _validate_graph_inputs_and_outputs(self):
    """Validates the inputs and outputs of a Graph Network."""
    # Check for redundancy in inputs.
    if len({id(i) for i in self.inputs}) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))

    for x in self.inputs:
      # Check that x has appropriate `_keras_history` metadata.
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Input tensors to a ' + cls_name + ' ' +
                         'must come from `tf.keras.Input`. '
                         'Received: ' + str(x) +
                         ' (missing previous layer metadata).')
      # Check that x is an input tensor.
      # pylint: disable=protected-access
      layer = x._keras_history.layer
      if len(layer._inbound_nodes) > 1 or (
          layer._inbound_nodes and layer._inbound_nodes[0].inbound_layers):
        cls_name = self.__class__.__name__
        logging.warning(cls_name + ' inputs must come from '
                        '`tf.keras.Input` (thus holding past layer metadata), '
                        'they cannot be the output of '
                        'a previous non-Input layer. '
                        'Here, a tensor specified as '
                        'input to "' + self.name + '" was not an Input tensor, '
                        'it was generated by layer ' + layer.name + '.\n'
                        'Note that input tensors are '
                        'instantiated via `tensor = tf.keras.Input(shape)`.\n'
                        'The tensor that caused the issue was: ' + str(x.name))
      if isinstance(x, ragged_tensor.RaggedTensor):
        self._supports_ragged_inputs = True

    # Check compatibility of batch sizes of Input Layers.
    input_batch_sizes = [
        training_utils.get_static_batch_size(x._keras_history.layer)
        for x in self.inputs
    ]
    consistent_batch_size = None
    for batch_size in input_batch_sizes:
      if batch_size is not None:
        if (consistent_batch_size is not None and
            batch_size != consistent_batch_size):
          raise ValueError('The specified batch sizes of the Input Layers'
                           ' are incompatible. Found batch sizes: {}'.format(
                               input_batch_sizes))
        consistent_batch_size = batch_size

    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors to a ' + cls_name + ' must be '
                         'the output of a TensorFlow `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))

  def _insert_layers(self, layers, relevant_nodes=None):
    layers = nest.flatten(layers)
    tf_utils.assert_no_legacy_layers(layers)
    node_to_depth = {}
    for depth, nodes in self._nodes_by_depth.items():
      node_to_depth.update({node: depth for node in nodes})
    # The nodes of these Layers that are relevant to this Network. If not
    # provided, assume all Nodes are relevant
    if not relevant_nodes:
      relevant_nodes = nest.flatten([layer._inbound_nodes for layer in layers])
    network_nodes = set(relevant_nodes + list(node_to_depth.keys()))

    def _get_min_depth(node):
      """Gets the minimum depth at which node can be computed."""
      min_depth = 0
      for layer, node_id, _, _ in node.iterate_inbound(include_arguments=True):
        inbound_node = layer._inbound_nodes[node_id]
        if inbound_node in node_to_depth:
          min_depth = min(min_depth, node_to_depth[inbound_node])
        elif inbound_node not in network_nodes:
          continue
        else:
          # Previous relevant nodes haven't been processed yet.
          return None
      # New node is one shallower than its shallowest input.
      return min_depth - 1

    # Insert nodes into `_nodes_by_depth` and other node attrs.
    unprocessed_nodes = copy.copy(relevant_nodes)
    i = 0
    while unprocessed_nodes:
      i += 1
      # Do a sanity check. This can occur if `Input`s from outside this Model
      # are being relied on.
      if i > 10000:
        raise ValueError('Layers could not be added due to missing '
                         'dependencies.')

      node = unprocessed_nodes.pop(0)
      depth = _get_min_depth(node)
      if depth is None:  # Defer until inbound nodes are processed.
        unprocessed_nodes.append(node)
        continue
      node_key = _make_node_key(node.outbound_layer.name,
                                node.outbound_layer._inbound_nodes.index(node))
      if node_key not in self._network_nodes:
        node_to_depth[node] = depth
        self._network_nodes.add(node_key)
        self._nodes_by_depth[depth].append(node)

    # Insert layers and update other layer attrs.
    layer_set = set(self._layers)
    deferred_layers = []
    for layer in layers:
      if layer not in layer_set:
        self._layers.append(layer)
        deferred_layers.append(layer)
        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)

        # This allows the added layer to broadcast mutations to the current
        # layer, which is necessary to ensure cache correctness.
        layer._attribute_sentinel.add_parent(self._attribute_sentinel)
        layer_set.add(layer)
    self._handle_deferred_layer_dependencies(deferred_layers)

    self._compute_tensor_usage_count()

  def _compute_tensor_usage_count(self):
    tensor_usage_count = collections.Counter()
    available_tensors = set(str(id(tensor)) for tensor in self.inputs)

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    depth_keys = depth_keys[1:]

    for depth in depth_keys:
      for node in self._nodes_by_depth[depth]:
        input_tensors = {
            str(id(tensor)) for tensor in nest.flatten(node.input_tensors)
        }
        if input_tensors.issubset(available_tensors):
          kwargs = copy.copy(node.arguments) if node.arguments else {}

          for tensor in nest.flatten(kwargs):
            if (isinstance(tensor,
                           (ops.Tensor, composite_tensor.CompositeTensor)) and
                hasattr(tensor, '_keras_history')):
              tensor_usage_count[str(id(tensor))] += 1

          for tensor in nest.flatten(node.input_tensors):
            tensor_usage_count[str(id(tensor))] += 1

          for output_tensor in nest.flatten(node.output_tensors):
            available_tensors.add(str(id(output_tensor)))

    for tensor in self.outputs:
      tensor_usage_count[str(id(tensor))] += 1

    self._tensor_usage_count = tensor_usage_count

  def _assert_weights_created(self):
    if self.dynamic:
      return
    if (not self._is_graph_network and
        'build' in self.__class__.__dict__ and
        not self.built):
      # For any model that has customized build() method but hasn't
      # been invoked yet, this will cover both sequential and subclass model.
      raise ValueError('Weights for model %s have not yet been created. '
                       'Weights are created when the Model is first called on '
                       'inputs or `build()` is called with an `input_shape`.' %
                       self.name)

  def _graph_network_add_loss(self, symbolic_loss):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [symbolic_loss])
    # Losses must be keyed on inputs no matter what in order to be supported in
    # DistributionStrategy.
    add_loss_layer = base_layer.AddLoss(
        unconditional=False, dtype=symbolic_loss.dtype)
    add_loss_layer(symbolic_loss)
    new_nodes.extend(add_loss_layer.inbound_nodes)
    new_layers.append(add_loss_layer)
    self._insert_layers(new_layers, new_nodes)

  def _graph_network_add_metric(self, value, aggregation, name):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [value])
    add_metric_layer = base_layer.AddMetric(
        aggregation, name, dtype=value.dtype)
    add_metric_layer(value)
    new_nodes.extend(add_metric_layer.inbound_nodes)
    new_layers.append(add_metric_layer)
    self._insert_layers(new_layers, new_nodes)

  @trackable.no_automatic_dependency_tracking
  def _set_save_spec(self, inputs):
    if self._saved_model_inputs_spec is not None:
      return  # Already set.

    input_names = self.input_names
    if not input_names:
      input_names = compile_utils.create_pseudo_input_names(inputs)

    flat_inputs = nest.flatten(inputs)
    specs = []
    for name, tensor in zip(input_names, flat_inputs):
      specs.append(
          tf_utils.get_tensor_spec(tensor, dynamic_batch=False, name=name))
    specs = nest.pack_sequence_as(inputs, specs)

    self._saved_model_inputs_spec = specs

  def _get_save_spec(self, dynamic_batch=True):
    if self._saved_model_inputs_spec is None:
      return None

    return nest.map_structure(
        lambda t: tf_utils.get_tensor_spec(t, dynamic_batch=dynamic_batch),
        self._saved_model_inputs_spec)

  @property
  def _trackable_saved_model_saver(self):
    return network_serialization.NetworkSavedModelSaver(self)
