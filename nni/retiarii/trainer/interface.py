import abc
import inspect
from ..model_apis.nn import add_record


class BaseTrainer(abc.ABC):
    """
    In this version, we plan to write our own trainers instead of using PyTorch-lightning, to
    ease the burden to integrate our optmization with PyTorch-lightning, a large part of which is
    opaque to us.

    We will try to align with PyTorch-lightning name conversions so that we can easily migrate to
    PyTorch-lightning in the future.

    Currently, our trainer = LightningModule + LightningTrainer. We might want to separate these two things
    in future.

    Trainer has a ``fit`` function with no return value. Intermediate results and final results should be
    directly sent via ``nni.report_intermediate_result()`` and ``nni.report_final_result()`` functions.
    """
    def __init__(self, *args, **kwargs):
        module = self.__class__.__module__
        if module is None or module == str.__class__.__module__:
            full_class_name = self.__class__.__name__
        else:
            full_class_name = module + '.' + self.__class__.__name__

        assert not kwargs
        argname_list = list(inspect.signature(self.__class__).parameters.keys())
        assert len(argname_list) == len(args), 'Error: {} not put input arguments in its super().__init__ function'.format(self.__class__)
        full_args = {}
        for i, arg_value in enumerate(args):
            if argname_list[i] == 'model':
                assert i == 0
                continue
            full_args[argname_list[i]] = args[i]
        add_record(id(self), {'modulename': full_class_name, 'args': full_args})

    @abc.abstractmethod
    def fit(self) -> None:
        pass
