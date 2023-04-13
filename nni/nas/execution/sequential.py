# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['SequentialExecutionEngine']

import logging
import time
import traceback
from typing import Iterable, List, cast
from typing_extensions import Literal

import nni
from nni.runtime.trial_command_channel import (
    set_default_trial_command_channel, get_default_trial_command_channel, TrialCommandChannel
)
from nni.nas.space import ExecutableModelSpace, ModelStatus
from nni.typehint import ParameterRecord, Parameters, TrialMetric

from .engine import ExecutionEngine
from .event import FinalMetricEvent, IntermediateMetricEvent, TrainingEndEvent

_logger = logging.getLogger(__name__)


class SequentialTrialCommandChannel(TrialCommandChannel):

    def __init__(self, engine: SequentialExecutionEngine, model: ExecutableModelSpace):
        self.engine = engine
        self.model = model

    def receive_parameter(self) -> ParameterRecord:
        return ParameterRecord(
            parameter_id=0,
            parameters=cast(Parameters, self.model),
        )

    def send_metric(
        self,
        type: Literal['PERIODICAL', 'FINAL'],  # pylint: disable=redefined-builtin
        parameter_id: int | None,
        trial_job_id: str,
        sequence: int,
        value: TrialMetric,
    ) -> None:
        if type == 'PERIODICAL':
            self.engine.dispatch_model_event(IntermediateMetricEvent(self.model, value))
        elif type == 'FINAL':
            self.engine.dispatch_model_event(FinalMetricEvent(self.model, value))
        else:
            raise ValueError(f'Unknown metric type: {type}')


class SequentialExecutionEngine(ExecutionEngine):
    """
    The execution engine will run every model in the current process.
    If multiple models have been submitted, they will be queued and run sequentially.

    Keyboard interrupt will terminate the currently running model and raise to let the main process know.
    """

    def __init__(self, max_model_count: int | None = None,
                 max_duration: float | None = None,
                 continue_on_failure: bool = False) -> None:
        super().__init__()

        self.max_model_count = max_model_count
        self.max_duration = max_duration
        self.continue_on_failure = continue_on_failure

        self._history: List[ExecutableModelSpace] = []
        self._model_count = 0
        self._total_duration = 0

    def _run_single_model(self, model: ExecutableModelSpace) -> None:
        model.status = ModelStatus.Training
        start_time = time.time()
        _prev_channel = get_default_trial_command_channel()
        try:
            # Reset the channel to overwrite get_next_parameter() and report_xxx_result()
            _channel = SequentialTrialCommandChannel(self, model)
            set_default_trial_command_channel(_channel)
            # Set the current parameter
            parameters = nni.get_next_parameter()
            assert parameters is model
            # Run training.
            model.execute()
            # Training success.
            status = ModelStatus.Trained
            duration = time.time() - start_time
            self._total_duration += duration
            _logger.debug('Execution time of model %d: %.2f seconds (total %.2f)',
                          self._model_count, duration, self._total_duration)
        except KeyboardInterrupt:
            # Training interrupted.
            duration = time.time() - start_time
            self._total_duration += duration
            _logger.error('Model %d is interrupted. Exiting gracefully...', self._model_count)
            status = ModelStatus.Interrupted
            raise
        except:
            # Training failed.
            _logger.error('Model %d fails to be executed.', self._model_count)
            duration = time.time() - start_time
            self._total_duration += duration
            status = ModelStatus.Failed
            if self.continue_on_failure:
                _logger.error(traceback.format_exc())
                _logger.error('Continue on failure. Skipping to next model.')
            else:
                raise
        finally:
            # Restore the trial command channel.
            set_default_trial_command_channel(_prev_channel)

            # Sometimes, callbacks could do heavy things here, e.g., retry the model.
            # So the callback should only be done at the very very end.
            # And we don't catch exceptions happen inside.
            self.dispatch_model_event(TrainingEndEvent(model, status))  # pylint: disable=used-before-assignment
            _logger.debug('Training end callbacks of model %d are done.', self._model_count)

    def submit_models(self, *models: ExecutableModelSpace) -> None:
        for model in models:
            if not model.status.frozen() or model.status.completed():
                raise RuntimeError(f'Model must be frozen before submitting, but got {model}')

            self._model_count += 1

            if self.max_model_count is not None and self._model_count > self.max_model_count:
                _logger.error('Maximum number of models reached (%d > %d). Models cannot be submitted anymore.',
                              self._model_count, self.max_model_count)
            if self.max_duration is not None and self._total_duration > self.max_duration:
                _logger.error('Maximum duration reached (%f > %f). Models cannot be submitted anymore.',
                              self._total_duration, self.max_duration)

            self._history.append(model)

            _logger.debug('Running model %d: %s', self._model_count, model)

            self._run_single_model(model)

    def list_models(self, status: ModelStatus | None = None) -> Iterable[ExecutableModelSpace]:
        if status is not None:
            return [m for m in self._history if m.status == status]
        return self._history

    def idle_worker_available(self) -> bool:
        """Return true because this engine will run models sequentially and never invokes this method when running the model."""
        return True

    def budget_available(self) -> bool:
        return (self.max_model_count is None or self._model_count < self.max_model_count) \
            and (self.max_duration is None or self._total_duration < self.max_duration)

    def shutdown(self) -> None:
        _logger.debug('Shutting down sequential engine.')

    def state_dict(self) -> dict:
        return {
            'model_count': self._model_count,
            'total_duration': self._total_duration,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict['model_count'] > 0:
            _logger.warning('Loading state for SequentialExecutionEngine does not recover previous submitted model history.')
        self._model_count = state_dict['model_count']
        self._total_duration = state_dict['total_duration']
