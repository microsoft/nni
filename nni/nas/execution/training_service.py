# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['TrainingServiceExecutionEngine']

import logging
import sys
import time
import weakref
from threading import Event, Thread
from typing import Iterable, TYPE_CHECKING, Any, cast

import nni
from nni.runtime.tuner_command_channel import command_type, TunerCommandChannel
from nni.typehint import TrialMetric
from nni.utils import MetricType

from nni.nas.space import ExecutableModelSpace, ModelStatus, GraphModelSpace

from .engine import ExecutionEngine
from .event import FinalMetricEvent, IntermediateMetricEvent, TrainingEndEvent

if TYPE_CHECKING:
    from nni.nas.experiment import NasExperiment

_logger = logging.getLogger(__name__)


class TrainingServiceExecutionEngine(ExecutionEngine):
    """
    The execution engine will submit every model onto training service.

    Resource management is implemented in this class.

    This engine doesn't include any optimization across graphs.

    NOTE: Due to the design of `nni.experiment`,
    the execution engine resorts to NasExperiment to submit trials as well as waiting for results.
    This is not ideal, because this engine might be one of the very few engines which need the training service.
    Ideally, the training service should be a part of the execution engine, not the experiment.

    Ideally, this class should not have any states. Its save and load methods should be empty.

    Parameters
    ----------
    nodejs_binding
        The nodejs binding of the experiment.
    fetch_intermediates
        Whether to fetch intermediate results from the training service when list models.
        Setting it to false for large-scale experiments can improve performance.
    """

    def __init__(self, nodejs_binding: NasExperiment, fetch_intermediates: bool = True) -> None:
        super().__init__()

        # NOTE: Currently the nodejs binding share the same instance as the experiment.
        # They will be separated in future iterations.
        self.nodejs_binding = nodejs_binding

        self.fetch_intermediates = fetch_intermediates

        # We need a model instance for callbacks.
        # They must be the exact same instance as the one submitted by the strategy.
        # The keys are parameter id (integers).
        self._models: dict[int, weakref.ReferenceType[ExecutableModelSpace]] = dict()

        # Submitted models might not appear in the database instantly.
        # We need a place to track them, in case they are accessed by `list_models()`.
        self._submitted_cache: dict[int, ExecutableModelSpace] = dict()

        # A counter for parameters. Currently, we can only guess it based on existing trials.
        # This should also be maintained at the nodejs side.
        self._current_parameter_id: int | None = None

        self._workers = 0

        # Connect to the tuner command channel.
        self._channel = TunerCommandChannel(nodejs_binding.tuner_command_channel)

        self._channel.on_initialize(self._initialize_callback)
        self._channel.on_request_trial_jobs(self._request_trial_jobs_callback)
        self._channel.on_report_metric_data(self._report_metric_data_callback)
        self._channel.on_trial_end(self._trial_end_callback)

        self._channel.connect()

        self._channel_listen_stop_event = Event()
        self._channel_listen_thread = Thread(
            target=self._channel.listen,
            kwargs={
                'stop_event': self._channel_listen_stop_event  # signal to stop this thread
            },
            daemon=True  # Set daemon to true: https://stackoverflow.com/a/58928213/6837658
        )
        self._channel_listen_thread.start()

        self._stopped = False

    def wait_models(self, *models: ExecutableModelSpace) -> None:
        """Wait models to finish training.

        If argument models is empty, wait for all models to finish.
        Using the experiment status as an indicator of all models' status,
        which is more efficient.

        For the models to receive status changes, the models must be the exact same instances as the ones submitted.
        Dumping and reloading the models, or retrieving the unsaved models from :meth:`list_models` won't work.
        """
        if not models:
            self._check_running()

            _logger.debug('Waiting for models. Using experiment status as an indicator of all models\' status.')

            training_model_patience = 0

            while True:
                status = self.nodejs_binding.get_status()
                if status in ['DONE', 'STOPPED', 'ERROR']:
                    # Nodejs status changed. Stop waiting.
                    # (even if there are UNKNOWN trials)
                    return

                # If no more trial is not set, the experiment will always show as running.
                # We need to check trial status statistics to determine whether the experiment is done.
                stats = self.nodejs_binding.get_job_statistics()
                training_models_found = False
                for stat in stats:
                    if self._interpret_trial_job_status(stat['trialJobStatus']) == ModelStatus.Training:
                        training_models_found = True
                        break

                if training_models_found:
                    if training_model_patience != 0:
                        _logger.debug('Running models found. Resetting patience. Current stats: %s', stats)
                        training_model_patience = 0
                else:
                    # The submit models could take up to 5 seconds to show up in the statistics.
                    _logger.debug('Waiting for running models to show up (patience: %d). Current stats: %s', training_model_patience, stats)
                    training_model_patience += 1
                    if training_model_patience > 6:
                        _logger.debug('No running models found. Assuming all models are done.')
                        return

                time.sleep(1)

        super().wait_models(*models)

    def submit_models(self, *models: ExecutableModelSpace) -> None:
        """Submit models to training service.

        See Also
        --------
        nni.nas.ExecutionEngine.submit_models
        """
        self._check_running()

        for model in models:
            if self._workers <= 0:
                # Use debug because concurrency is ignored in many built-in strategies.
                _logger.debug('Submitted models exceed concurrency. Remaining concurrency is %d.', self._workers)

            parameter_id = self._next_parameter_id()
            self._models[parameter_id] = weakref.ref(model)
            self._submitted_cache[parameter_id] = model

            placement = None
            if isinstance(model, GraphModelSpace):
                placement = model.export_placement_constraint()

            self._channel.send_trial(
                parameter_id=parameter_id,
                parameters=cast(Any, model),
                placement_constraint=placement
            )

            model.status = ModelStatus.Training

            self._workers -= 1
            _logger.debug('Submitted model with parameter id %d. Remaining resource: %d.', parameter_id, self._workers)

    def list_models(self, status: ModelStatus | None = None) -> Iterable[ExecutableModelSpace]:
        """Retrieve models previously submitted.

        To support a large-scale experiments with thousands of trials,
        this method will retrieve the models from the nodejs binding (i.e., from the database).
        The model instances will be re-created on the fly based on the data from database.
        Although they are the same models semantically, they might not be the same instances.
        Exceptions are those still used by the strategy.
        Their weak references are kept in the engine and thus the exact same instances are returned.

        Parameters
        ----------
        status
            The status of the models to be retrieved.
            If None, all models will be retrieved.
        include_intermediates
            Whether to include intermediate models.
        """
        self._check_running()

        for trial in self.nodejs_binding.list_trial_jobs():
            if len(trial.hyperParameters) != 1:
                # This trial is not submitted by the engine.
                _logger.warning('Found trial "%s" with unexpected number of parameters. '
                                'It may not be submitted by the engine. Skip.', trial.trialJobId)
                continue

            param = trial.hyperParameters[0]
            parameter_id = param.parameter_id
            model = self._find_reference_model(parameter_id)  # type: ignore

            # Check model status first to avoid loading the unneeded models.
            if model is not None:
                model_status = model.status
            else:
                model_status = self._interpret_trial_job_status(trial.status)
            if status is not None and model_status != status:
                continue

            if model is None:
                # The model has been garbage-collected.
                # Load it from trial parameters.

                # The hyper-parameter is incorrectly loaded at NNI manager.
                # Dump and reload it here will turn it into a model.
                model: ExecutableModelSpace = nni.load(nni.dump(param.parameters))
                if not isinstance(model, ExecutableModelSpace):
                    _logger.error('The parameter of trial "%s" is not a model. Skip.', trial.trialJobId)
                    continue

                model.status = model_status
                if trial.finalMetricData:
                    if len(trial.finalMetricData) != 1:
                        _logger.warning('The final metric data of trial "%s" is not a single value. Taking the last one.',
                                        trial.trialJobId)
                    # The data has already been unpacked at the binding.
                    model.metrics.final = cast(TrialMetric, trial.finalMetricData[-1].data)

                if self.fetch_intermediates:
                    metrics = self.nodejs_binding.get_job_metrics(trial.trialJobId)
                    for metric_data in metrics.get(trial.trialJobId, []):
                        if metric_data.type == 'PERIODICAL':
                            model.metrics.add_intermediate(metric_data.data)

            yield model

        # Yield models still in the cache, but not in the database.
        # This is to support the case where users call `list_models` instantly after submit models.
        # Models might take seconds to show up in the database.
        for model in self._submitted_cache.values():
            if status is None or model.status == status:
                yield model

    def idle_worker_available(self) -> bool:
        """Return the number of available resources.

        The resource is maintained by the engine itself.
        It should be fetched from nodejs side directly in future.
        """
        return self._workers > 0

    def budget_available(self) -> bool:
        """Infer the budget from resources.

        This should have a dedicated implementation on the nodejs side in the future.
        """
        self._check_running()

        return self.nodejs_binding.get_status() in ['INITIALIZED', 'RUNNING', 'TUNER_NO_MORE_TRIAL']

    def shutdown(self) -> None:
        self._stopped = True
        # Stop the inner command listening thread (if it hasn't stopped).
        self._channel_listen_stop_event.set()
        # This shutdown needs to work together with (actually after) the shutdown on the other side,
        # otherwise it's possible that the thread is still waiting for a command that is never coming.
        self._channel_listen_thread.join()

    def load_state_dict(self, state_dict: dict) -> None:
        _logger.info('Loading state for training service engine does nothing.')

    def state_dict(self) -> dict:
        return {}

    # Callbacks for incoming commands. This will be run in another thread.

    def _initialize_callback(self, command: command_type.Initialize) -> None:
        self._channel.send_initialized()

    def _request_trial_jobs_callback(self, command: command_type.RequestTrialJobs) -> None:
        self._workers += command.count
        _logger.debug('New resources received. Remaining resource: %d.', self._workers)

    def _report_metric_data_callback(self, command: command_type.ReportMetricData) -> None:
        model = self._find_reference_model(command.parameter_id)
        # If model is none, strategy must have thrown the model away.
        # Why should I care if strategy stops caring about the model?
        # It can be retrieved from `list_models()` anyway.
        if model is not None:
            if command.type == MetricType.PERIODICAL:
                self.dispatch_model_event(IntermediateMetricEvent(model, cast(TrialMetric, command.value)))
            elif command.type == MetricType.FINAL:
                self.dispatch_model_event(FinalMetricEvent(model, cast(TrialMetric, command.value)))
            else:
                raise ValueError('Unknown metric type: %r' % command.type)
        else:
            _logger.debug('Received metric data of "%s" (parameter id: %d) but the model has been garbage-collected. Skip.',
                          command.trial_job_id, command.parameter_id)

    def _trial_end_callback(self, command: command_type.TrialEnd) -> None:
        if len(command.parameter_ids) != 1:
            _logger.warning('Received trial end event of "%s" with unexpected number of parameters. '
                            'It may not be submitted by the engine. Skip.', command.trial_job_id)
        else:
            # The trial must have exactly one parameter.
            model = self._find_reference_model(command.parameter_ids[0])
            # Similar to ReportMetricData.
            if model is not None:
                model_status = self._interpret_trial_job_status(command.event)
                self.dispatch_model_event(TrainingEndEvent(model, model_status))
            else:
                _logger.debug('Received trial end event of "%s" (parameter id: %d) but the model has been garbage-collected. Skip.',
                              command.trial_job_id, command.parameter_ids[0])

    # Ignore other commands.

    def _check_running(self) -> None:
        if self._stopped:
            raise RuntimeError('The engine has been stopped. Cannot take any more action.')

    def _next_parameter_id(self) -> int:
        """Get the next available parameter id.

        Communicate with nodejs binding if necessary.
        """
        if self._current_parameter_id is None:
            # Get existing trials and infer the next parameter id.
            trials = self.nodejs_binding.list_trial_jobs()
            existing_ids = [param.parameter_id for trial in trials for param in trial.hyperParameters]
            self._current_parameter_id = max(existing_ids) if existing_ids else -1

        self._current_parameter_id += 1
        return self._current_parameter_id

    def _find_reference_model(self, parameter_id: int) -> ExecutableModelSpace | None:
        """Retrieve the reference model by a parameter id.

        The reference model is the model instance submitted by the strategy.
        It is used to create a new model instance based on the information provided by the nodejs binding.
        """
        # The model is considered stored on the nodejs side.
        # Save to invalidate the cache.
        self._invalidate_submitted_cache(parameter_id)

        if parameter_id in self._models:
            model = self._models[parameter_id]()
            if model is not None:
                return model
            _logger.debug('The reference model for parameter "%d" has been garbage-collected. Removing it from cache.',
                          parameter_id)
            self._models.pop(parameter_id)
        return None

    def _invalidate_submitted_cache(self, parameter_id: int) -> None:
        """Remove the cache item when the parameter id has been found in the database of NNI manager."""
        self._submitted_cache.pop(parameter_id, None)

    def _interpret_trial_job_status(self, status: str) -> ModelStatus:
        """Translate the trial job status into a model status."""
        if status in ['WAITING', 'RUNNING', 'UNKNOWN']:
            return ModelStatus.Training
        if status == 'SUCCEEDED':
            return ModelStatus.Trained
        return ModelStatus.Failed


# For trial end #

def trial_entry() -> None:
    """The entry point for the trial job launched by training service."""
    params = nni.get_next_parameter()
    assert isinstance(params, ExecutableModelSpace), 'Generated parameter should be an ExecutableModelSpace.'
    params.execute()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python -m nni.nas.execution.training_service trial', file=sys.stderr)
        sys.exit(1)
    if sys.argv[1] == 'trial':
        # Start a trial job.
        trial_entry()
