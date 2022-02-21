_import_failed = False
try:
    from nni.retiarii.oneshot.pytorch.strategy import (
        DartsModule, ProxylessModule, SNASModule, EnasModule, RandomSamplingModule,
        InterleavedTrainValDataLoader
    )
except ImportError:
    _import_failed = True
