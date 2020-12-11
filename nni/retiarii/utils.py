import inspect

def import_(target: str, allow_none: bool = False) -> 'Any':
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)

'''# legacy
class TraceClassArguments:
    def __init__(self):
        self.recorded_arguments = None
    
    def __enter__(self):
        enable_record_args()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            # return False # uncomment to pass exception through
        self.recorded_arguments = get_records()
        disable_record_args()'''

_records = {}

def get_records():
    global _records
    return _records

def add_record(key, value):
    """
    """
    global _records
    if _records is not None:
        assert key not in _records, '{} already in _records'.format(key)
        _records[key] = value

def _register_module(original_class):
    print('zql: ', original_class.__name__)

    orig_init = original_class.__init__
    argname_list = list(inspect.signature(original_class).parameters.keys())
    # Make copy of original __init__, so we can call it without recursion

    def __init__(self, *args, **kws):
        full_args = {}
        full_args.update(kws)
        for i, arg in enumerate(args):
            full_args[argname_list[i]] = args[i]
        add_record(id(self), full_args)

        orig_init(self, *args, **kws) # Call the original __init__

    original_class.__init__ = __init__ # Set the class' __init__ to the new one
    return original_class

    '''if not inspect.isclass(module_class):
        raise TypeError('module must be a class, '
                        f'but got {type(module_class)}')

    if module_name is None:
        module_name = module_class.__name__
    if not force and module_name in self._module_dict:
        raise KeyError(f'{module_name} is already registered '
                        f'in {self.name}')
    self._module_dict[module_name] = module_class'''

def register_module():
    """
    Register a module.
    """

    # use it as a decorator: @register_module()
    def _register(cls):
        m = _register_module(
            original_class=cls)
        return m

    return _register
