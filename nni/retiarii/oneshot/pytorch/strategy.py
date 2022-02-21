"""Strategy integration of one-shot.

This file is put here simply because it relies on "pytorch".
For consistency, please consider importing strategies from ``nni.retiarii.strategy``.
For example, ``nni.retiarii.strategy.DartsStrategy`` (this requires pytorch to be installed of course).
"""

from nni.retiarii.graph import Evaluator
from nni.retiarii.strategy.base import BaseStrategy


class OneShotStrategy(BaseStrategy):

    def __init__(self, oneshot_module, **kwargs):
        self.oneshot_module = oneshot_module
        self.oneshot_kwargs = kwargs

    def get_dataloader(self, evaluator: Evaluator):
        """
        One-shot strategy typically requires a customized dataloader.
        
        Parameters
        ----------
        evaluator : Evaluator
        """
        raise NotImplementedError()

    def run(self, base_model, applied_mutators):
        # one-shot strategy doesn't use ``applied_mutators``
        # but get the "mutators" on their own

        _reason = 'The reason might be that you have used the wrong execution engine. Try to set engine to `oneshot` and try again.'

        if applied_mutators:
            raise ValueError('Mutator is not empty. ' + _reason)
        

        model = self.oneshot_module(base_model, **self.oneshot_kwargs)
        dataloader = self.get_dataloader(evaluator)
        evaluator.trainer.fit(model, dataloader)


class DartsStrategy(OneShotStrategy):
    def get_dataloader(self, evaluator: Evaluator):
        InterleavedTrainValDataLoader(cls.train_dataloader, cls.val_dataloaders)