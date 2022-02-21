from .base import BaseStrategy

_import_failed = False

try:
    from nni.retiarii.oneshot.pytorch.strategy import DartsStrategy
except ImportError:
    _import_failed = True

    class ImportFailedStrategy(BaseStrategy):
        def run(self, base_model, applied_mutators):
            raise

    class DartsStrategy
