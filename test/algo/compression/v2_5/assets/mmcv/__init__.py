try:
    from .detection import *
except ImportError:
    raise ImportError('Please install `mmdet` in dev mode!')
