# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Wrappers of HPO tuners as NAS strategy."""

__all__ = ['HPOTunerWrapper', 'TPE']

import logging
import time
import threading

from .base import Strategy

import nni
from nni.nas.execution import ExecutionEngine
from nni.nas.execution.event import FinalMetricEvent, TrainingEndEvent, ModelEventType
from nni.nas.space import ExecutableModelSpace, ModelStatus
from nni.tuner import Tuner

_logger = logging.getLogger(__name__)


class HPOTunerStrategy(Strategy):
    """
    Wrap a HPO tuner as a NAS strategy.

    Currently we only support:

    - Search space with choices.
    - Calling the tuner's ``generate_parameters`` method to generate new models.
    - Calling the tuner's ``receive_trial_result`` method to report the results of models.

    We don't support advanced features like checkpointing, resuming, or customized trials.

    Parameters
    ----------
    tuner
        A HPO tuner.
    """

    def __init__(self, tuner: Tuner):
        super().__init__()

        self.tuner = tuner

        # Tuner is not thread safe. We need to lock the tuner when calling its methods.
        self._thread_lock = threading.Lock()

        self._model_count = 0
        self._model_to_id: dict[ExecutableModelSpace, int] = {}

    def extra_repr(self) -> str:
        return f'tuner={self.tuner!r}'

    def _initialize(self, model_space: ExecutableModelSpace, engine: ExecutionEngine) -> ExecutableModelSpace:
        engine.register_model_event_callback(ModelEventType.FinalMetric, self.on_metric)
        engine.register_model_event_callback(ModelEventType.TrainingEnd, self.on_training_end)
        return model_space

    def _cleanup(self) -> None:
        self.engine.unregister_model_event_callback(ModelEventType.FinalMetric, self.on_metric)
        self.engine.unregister_model_event_callback(ModelEventType.TrainingEnd, self.on_training_end)

    def _run(self) -> None:
        tuner_search_space = {label: mutable.as_legacy_dict() for label, mutable in self.model_space.simplify().items()}
        _logger.debug('Tuner search space: %s', tuner_search_space)

        with self._thread_lock:
            self.tuner.update_search_space(tuner_search_space)

        while self.engine.budget_available():
            if self.engine.idle_worker_available():
                with self._thread_lock:
                    try:
                        param = self.tuner.generate_parameters(self._model_count)
                    except nni.NoMoreTrialError:
                        _logger.warning('No more trial generated by tuner. Stopping.')
                        break
                    _logger.debug('[%4d] Tuner generated parameters: %s', self._model_count, param)
                    model = self.model_space.freeze(param)
                    self._model_to_id[model] = self._model_count
                    self._model_count += 1

                self.engine.submit_models(model)

            time.sleep(1.)

    def on_metric(self, event: FinalMetricEvent) -> None:
        with self._thread_lock:
            model_id = self._model_to_id[event.model]
            self.tuner.receive_trial_result(model_id, event.model.sample, event.metric)

    def on_training_end(self, event: TrainingEndEvent) -> None:
        with self._thread_lock:
            model_id = self._model_to_id.pop(event.model)
            self.tuner.trial_end(model_id, event.status == ModelStatus.Trained)

    def load_state_dict(self, state_dict: dict) -> None:
        self._model_count = state_dict.get('model_count', 0)
        if self._model_count > 0:
            _logger.warning('Loaded %d previously submitted models, but they are not recorded, or reported to tuner.')

    def state_dict(self) -> dict:
        return {'model_count': self._model_count}


class TPE(HPOTunerStrategy):
    """The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach.

    Find the details in
    `Algorithms for Hyper-Parameter Optimization <https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>`__.

    SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements,
    and then subsequently choose new hyperparameters to test based on this model.
    """

    def __init__(self):
        from nni.algorithms.hpo.tpe_tuner import TpeTuner
        super().__init__(TpeTuner())


# alias for backward compatibility
TPEStrategy = TPE
