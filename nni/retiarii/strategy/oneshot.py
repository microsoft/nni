from .base import BaseStrategy

try:
    from nni.retiarii.oneshot.pytorch.strategy import DartsStrategy
except ImportError as import_err:
    _import_err = import_err

    class ImportFailedStrategy(BaseStrategy):
        def run(self, base_model, applied_mutators):
            raise _import_err

    # otherwise typing check will pointing to the wrong location
    globals()['DartsStrategy'] = ImportFailedStrategy
