import functools
import weakref

from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import func_graph, ops
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, resource_variable_ops
from tensorflow.python.util import object_identity, tf_decorator

from . import lift_to_graph

class Function:
    def __init__(self, python_function, name, input_signature=None, autograph=True):
        self._python_function = python_function
        self._function_spec = function_lib.FunctionSpec.from_function_and_signature(python_function, input_signature)
        self._autograph = autograph
        self._created_variables = None
        self._stateful_fn = None
        self._stateless_fn = None
        self._descriptor_cache = weakref.WeakKeyDictionary()
        self._name = name
        self._input_signature = input_signature
        self._call_counter = _CallCounter(10)


    def _defun_with_scope(self, scope):
        weak_wrapped_fn = None
        def wrapped_fn(*args, **kwargs):
            with ops.get_default_graph()._variable_creator_scope(scope, priority=50):
                return weak_wrapped_fn().__wrapped__(*args, **kwargs)
        weak_wrapped_fn = weakref.ref(wrapped_fn)
        return self._defun(tf_decorator.make_decorator(self._python_function, wrapped_fn))


    def _defun(self, fn):
        attributes = {}
        if not attributes:
            attributes = None
        return function_lib.defun_with_attributes(
            fn,
            input_signature=self.input_signature,
            attributes=attributes,
            autograph=self._autograph
        )


    def _initialize(self, args, kwargs, add_initializers_to=None):
        created_variables = []
        lifted_initializer_graph = func_graph.FuncGraph('initializer')

        def variable_capturing_scope(unused_next_creator, **kwargs):
            v = tf.eager.def_function.UnliftedInitializerVariable(
                add_initializers_to=add_initializers_to,
                lifted_initializer_graph=lifted_initializer_graph,
                **kwargs
            )
            created_variables.append(weakref.ref(v))
            return v

        self._created_variables = created_variables
        self._stateful_fn = self._defun_with_scope(variable_capturing_scope)
        self._stateful_fn._name = self._name
        self._lifted_initializer_graph = lifted_initializer_graph
        self._graph_deleter = _FunctionDeleter(self._lifted_initializer_graph)
        self._concrete_stateful_fn = self._stateful_fn._get_concrete_function_internal_garbage_collected(*args, **kwargs)

        def invalid_creator_scope(*unused_args, **unused_kwargs):
            raise ValueError('invalid creator scope')

        self._stateless_fn = self._defun_with_scope(invalid_creator_scope)
        self._stateless_fn._name = self._name


    def _get_tracing_count(self):
        result = self._stateless_fn.tracing_count if self._stateless_fn else 0
        result += self._stateful_fn.tracing_count if self._stateful_fn else 0
        return result


    def __call__(self, *args, **kwargs):
        tracing_count = self._get_tracing_count()
        result = self._call(*args, **kwargs)
  
        if tracing_count == self._get_tracing_count():
            self._call_counter.called_without_tracing()
            return result
    
        self._call_counter.called_with_tracing()
        recent_tracing_count = self._call_counter.get_tracing_count()
        if recent_tracing_count >= 5:
            print('large tracing count:', recent_tracing_count)
        return result


    def _call(self, *args, **kwargs):
        if self._created_variables:
            return self._stateless_fn(*args, **kwargs)
        elif self._stateful_fn is not None:
            results = self._stateful_fn(*args, **kwargs)
            if self._created_variables:
                raise ValueError('run-time creating variable')
            return results

        initializers = []
        self._initialize(args, kwargs, add_initializers_to=initializers)

        lifted = False
        if self._created_variables:
            try:
                self._initialize_uninitialized_variables(initializers)
                lifted = True
            except lift_to_graph.UnliftableError:
                pass
            if lifted:
                return self._stateless_fn(*args, **kwargs)
        else:
            canon_args, canon_kwargs = self._stateful_fn._function_spec.canonicalize_function_inputs(*args, **kwargs)
            return self._concrete_stateful_fn._filtered_call(canon_args, canon_kwargs)

        def fn_with_cond(*inner_args, **inner_kwargs):
            condition = True
            for wr in self._created_variables:
                variable = wr()
                if variable is None:
                    raise ValueError('variable is garbage-collected')
                condition = math_ops.logical_and(condition, resource_variable_ops.var_is_initialized_op(variable.handle))
            return control_flow_ops.cond(
                condition,
                lambda: self._stateless_fn(*inner_args, **inner_kwargs),
                functools.partial(self._concrete_stateful_fn._filtered_call, inner_args, inner_kwargs)
            )

        canon_args, canon_kwargs = self._stateful_fn._function_spec.canonicalize_function_inputs(*args, **kwargs)
        return function_lib.defun(fn_with_cond)(*canon_args, **canon_kwargs)


    @property
    def input_signature(self):
        return self._function_spec.input_signature


    def _initialize_uninitialized_variables(self, initializers):
        if not initializers:
            return

        @function_lib.defun(autograph=False)
        def initialize_variables():
            op_map = object_identity.ObjectIdentityDictionary()
            with ops.init_scope():
                var_is_initialized = []
                for v, _ in initializers:
                    var_is_initialized.append(resource_variable_ops.var_is_initialized_op(v.handle))
                var_is_initialized = array_ops.stack(var_is_initialized).numpy()

            inits = []
            for (v, init), is_initialized in zip(initializers, var_is_initialized):
                with ops.init_scope():
                    if is_initialized:
                        continue
                inits.append(init)

            if inits:
                op_map = lift_to_graph.lift_to_graph(inits, ops.get_default_graph(), op_map=op_map)
            for (v, init), is_initialized in zip(initializers, var_is_initialized):
                with ops.init_scope():
                    if is_initialized:
                        continue
                v.assign(op_map[init], read_value=False)

        with ops.init_scope():
            return initialize_variables.get_concrete_function()()


    def _get_concrete_function_garbage_collected(self, *args, **kwargs):
        if self._stateful_fn is None:
            initializers = []
            self._initialize(args, kwargs, add_initializers_to=initializers)
            self._initialize_uninitialized_variables(initializers)

        if self._created_variables:
            return self._stateless_fn._get_concrete_function_garbage_collected(*args, **kwargs)
        elif self._stateful_fn is not None:
            concrete = self._stateful_fn._get_concrete_function_garbage_collected(*args, **kwargs)
            if self._created_variables:
                raise ValueError('run-time creating variable')
            return concrete


    def get_concrete_function(self, *args, **kwargs):
        concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
        concrete._garbage_collector.release()
        return concrete


    def __get__(self, instance, owner):
        del owner
        if instance not in self._descriptor_cache:
            if instance is None:
                return self
            self._descriptor_cache[instance] = function_lib.class_method_to_instance_method(self, instance)
        return self._descriptor_cache[instance]


class _CallCounter:
    def __init__(self, max_call_history):
        self._max_call_history = max_call_history
        self._calls_per_tracings = []
        self.call_count = 0

    def called_with_tracing(self):
        self.call_count += 1
        self._calls_per_tracings.append(1)
        while self._calls_per_tracings:
            if self.call_count - self._calls_per_tracings[0] <= self._max_call_history:
                break
            self.call_count -= self._calls_per_tracings.pop(0)

    def called_without_tracing(self):
        if not self._calls_per_tracings:
            self._calls_per_tracings = [0]
        self._calls_per_tracings[-1] += 1
        self.call_count += 1

    def get_tracing_count(self):
        return len(self._calls_per_tracings)


class _FunctionDeleter:
    def __init__(self, func):
        self._func = func

    def __del__(self):
        try:
            func_graph.dismantle_func_graph(self._func)
        except:
            pass


def function(func):
    try:
        name = func.__name__
    except AttributeError:
        name = 'function'
    return Function(func, name)
