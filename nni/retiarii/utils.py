import traceback
from .nn.pytorch import enable_record_args, get_records, disable_record_args

def import_(target: str, allow_none: bool = False) -> 'Any':
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)

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
        disable_record_args()
