# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['NasExperiment']

import atexit
import logging
import warnings
from pathlib import Path
from typing import Any, ClassVar, cast
from typing_extensions import Literal

import nni
from nni.experiment import Experiment, RunMode
from nni.nas.evaluator import Evaluator
from nni.nas.execution import ExecutionEngine, TrainingServiceExecutionEngine, SequentialExecutionEngine
from nni.nas.space import ExecutableModelSpace, BaseModelSpace, GraphModelSpace
from nni.nas.strategy import Strategy
from nni.nas.utils.serializer import get_default_serializer
from nni.tools.nnictl.config_utils import Experiments
from .config import (
    NasExperimentConfig, ExecutionEngineConfig,
    TrainingServiceEngineConfig, CgoEngineConfig, SequentialEngineConfig,
    ModelFormatConfig, GraphModelFormatConfig, SimplifiedModelFormatConfig, RawModelFormatConfig
)

_logger = logging.getLogger(__name__)


def _get_current_timestamp() -> int:
    import time
    return int(time.time() * 1000)


class NasExperiment(Experiment):
    """
    The entry for a NAS experiment.
    Users can use this class to start/stop or inspect an experiment, like exporting the results.

    Experiment is a sub-class of :class:`nni.experiment.Experiment`, there are many similarities such as
    configurable training service to distributed running the experiment on remote server.
    But unlike :class:`nni.experiment.Experiment`, :class:`NasExperiment` doesn't support configure:

    - ``trial_code_directory``, which can only be current working directory.
    - ``search_space``, which is auto-generated in NAS.
    - ``trial_command``, which is auto-set to launch the modulized trial code.

    :class:`NasExperiment` also doesn't have tuner/assessor/advisor, because such functionality is already implemented in strategy.

    Also, unlike :class:`nni.experiment.Experiment` which is bounded to a node server,
    :class:`NasExperiment` optionally starts a node server to schedule the trials, depending on the configuration of execution engine.
    When the strategy is one-shot, the step of launching node server is omitted, and the experiment is run locally by default.

    Configurations of experiments, such as execution engine, number of GPUs allocated,
    should be put into a :class:`NasExperimentConfig` and passed to the initialization of an experiment.
    The config can be also altered after the experiment is initialized.

    Parameters
    ----------
    model_space
        The model space to search.
    evaluator
        Evaluator for the experiment.
    strategy
        Exploration strategy. Can be multi-trial or one-shot.
    config
        Configurations of the experiment. See :class:`~nni.nas.experiment.NasExperimentConfig` for details.
        When not provided, a default config will be created based on current model space, evaluator and strategy.
        Detailed rules can be found in :meth:`nni.nas.experiment.NasExperimentConfig.default`.

    Warnings
    --------
    ``wait_completion`` doesn't work for NAS experiment because NAS experiment **always wait for completion**.

    Examples
    --------
    >>> base_model = MobileNetV3Space()
    >>> search_strategy = strategy.Random()
    >>> model_evaluator = Classification()
    >>> exp = NasExperiment(base_model, model_evaluator, search_strategy)
    >>> exp_config.max_trial_number = 20
    >>> exp_config.training_service.use_active_gpu = False
    >>> exp.run(exp_config, 8081)

    Export top models and re-initialize the top model:

    >>> for model_dict in exp.export_top_models(formatter='dict'):
    ...     print(model_dict)
    >>> with model_context(model_dict):
    ...     final_model = Net()
    """

    config: NasExperimentConfig

    _state_dict_version: ClassVar[int] = 1

    def __init__(self,
                 model_space: BaseModelSpace,
                 evaluator: Evaluator,
                 strategy: Strategy,
                 config: NasExperimentConfig | None = None,
                 id: str | None = None) -> None:  # pylint: disable=redefined-builtin
        self.model_space = model_space
        self.evaluator = evaluator
        self.strategy = strategy

        super().__init__(config or NasExperimentConfig.default(model_space, evaluator, strategy), id=id)

        # TODO: check the engine, strategy, and evaluator are compatible with each other.

        # NOTE:
        # I didn't put model space conversion and engine creation here because
        # users might update the config after creating the experiment via `exp.config`,
        # and I don't want to disallow that, or create the engine twice.
        #
        # One potential problem is that checkpoints might not be correctly saved
        # in tricky states (e.g., before running or after stopping),
        # but I guessed they are not intended use cases.

        # These attributes are only used when NNI manager is available.
        # They will be initialized in run().
        self._tuner_command_channel: str | None = None

        self._engine: ExecutionEngine | None = None
        self._exec_model_space: ExecutableModelSpace | None = None

    @property
    def tuner_command_channel(self) -> str:
        if self._tuner_command_channel is None:
            raise RuntimeError('Experiment is not running.')
        return self._tuner_command_channel

    def execution_engine_factory(self, config: ExecutionEngineConfig) -> ExecutionEngine:
        if isinstance(config, TrainingServiceEngineConfig):
            return TrainingServiceExecutionEngine(self)
        elif isinstance(config, CgoEngineConfig):
            from nni.experiment.config.training_services import RemoteConfig
            from nni.nas.execution.cgo import CrossGraphOptimization
            engine = TrainingServiceExecutionEngine(self)
            assert isinstance(config.training_service, RemoteConfig)
            cgo_middleware = CrossGraphOptimization(
                config.training_service,
                config.max_concurrency_cgo,
                config.batch_waiting_time
            )
            cgo_middleware.set_engine(engine)
            return cgo_middleware
        elif isinstance(config, SequentialEngineConfig):
            return SequentialExecutionEngine(config.max_model_count, config.max_duration, config.continue_on_failure)
        else:
            raise ValueError(f'Unsupported engine config: {config}')

    def executable_model_factory(self, config: ModelFormatConfig) -> ExecutableModelSpace:
        from nni.nas.nn.pytorch import ModelSpace
        if not isinstance(self.model_space, ModelSpace):
            raise TypeError('Model space must inherit ModelSpace and also be a PyTorch model.')
        if isinstance(config, GraphModelFormatConfig):
            from nni.nas.space.pytorch import PytorchGraphModelSpace
            return PytorchGraphModelSpace.from_model(self.model_space, self.evaluator, config.dummy_input)
        elif isinstance(config, SimplifiedModelFormatConfig):
            from nni.nas.space import SimplifiedModelSpace
            return SimplifiedModelSpace.from_model(self.model_space, self.evaluator)
        elif isinstance(config, RawModelFormatConfig):
            from nni.nas.space import RawFormatModelSpace
            return RawFormatModelSpace.from_model(self.model_space, self.evaluator)
        else:
            raise ValueError(f'Unsupported model format config: {config}')

    def _start_nni_manager(self, port: int, debug: bool, run_mode: RunMode = RunMode.Background,
                           tuner_command_channel: str | None = None, tags: list[str] = []) -> None:
        """Manually set tuner command channel before starting NNI manager,
        because the other side (client side) of the tuner command channel is not launched by NNI manager.
        """
        if tuner_command_channel is None:
            tuner_command_channel = f'ws://localhost:{port}/tuner'
            _logger.debug('Tuner command channel is set to: %s', tuner_command_channel)

        self._tuner_command_channel = tuner_command_channel

        return super()._start_nni_manager(port, debug, run_mode, tuner_command_channel, tags + ['retiarii'])

    def _start_without_nni_manager(self):
        """Write the current experiment to experiment manifest.

        This is mock what has been done in launcher and nnimanager.
        """
        Experiments().add_experiment(
            self.id,
            'N/A',
            _get_current_timestamp(),
            'N/A',
            self.config.experiment_name,
            'N/A',
            status='RUNNING',
            tag=['retiarii'],
            logDir=str(self.config.experiment_working_directory)
        )

        # TODO: link the _latest symlink here.

    def _stop_without_nni_manager(self):
        """Update the experiment manifest.

        For future resume and experiment management.
        """
        Experiments().update_experiment(self.id, 'status', 'STOPPED')
        Experiments().update_experiment(self.id, 'endTime', _get_current_timestamp())

    def _start_engine_and_strategy(self) -> None:
        config = self.config.canonical_copy()

        _logger.debug('Creating engine and initializing strategy...')
        self._engine = self.execution_engine_factory(config.execution_engine)
        self._exec_model_space = self.executable_model_factory(config.model_format)
        self.strategy.initialize(self._exec_model_space, self._engine)

        if self._action == 'resume':
            self.load_checkpoint()

        if self._nni_manager_required():
            self._send_space_to_nni_manager()

        _logger.debug('Saving checkpoint before starting experiment...')
        self.save_checkpoint()

        _logger.info('Experiment initialized successfully. Starting exploration strategy...')

        self.strategy.run()

    def start(self, port: int = 8080, debug: bool = False, run_mode: RunMode = RunMode.Background) -> None:
        """Start a NAS experiment.

        Since NAS experiments always have strategies running in main thread,
        :meth:`start` will not exit until the strategy finishes its run.

        ``port`` and ``run_mode`` are only meaningful when :meth:`_nni_manager_required` returns true.

        Parameters
        ----------
        port
            Port to start NNI manager.
        debug
            If true, logging will be in debug mode.
        run_mode
            Whether to have the NNI manager in background, or foreground.

        See Also
        --------
        nni.experiment.Experiment.start
        """

        if run_mode is not RunMode.Detach:
            atexit.register(self.stop)

        self._start_logging(debug)

        if self._nni_manager_required():
            _logger.debug('Starting NNI manager...')
            if run_mode != RunMode.Background and self._action in ['create', 'resume']:
                _logger.warning('Note that run_mode in NAS will only change the behavior of NNI manager. '
                                'Strategy will still run in main thread.')
            self._start_nni_manager(port, debug, run_mode)
        else:
            _logger.debug('Writing experiment to manifest...')
            self._start_without_nni_manager()

        if self._action in ['create', 'resume']:
            self._start_engine_and_strategy()

    def stop(self) -> None:
        """Stop a NAS experiment.
        """
        _logger.info('Stopping experiment, please wait...')
        atexit.unregister(self.stop)

        self.save_checkpoint()

        if self._nni_manager_required():
            _logger.debug('Stopping NNI manager...')
            self._stop_nni_manager()
            # NNI manager should be stopped before engine.
            # Training service engine need the terminate signal so that the listener thread can be stopped.
        else:
            _logger.debug('Updating experiment manifest...')
            self._stop_without_nni_manager()

        # NOTE: Engine is designed to be disposable.
        # It should never restart because one experiment can't run twice.
        if self._engine is not None:
            self._engine.shutdown()

        _logger.debug('Stopping logging...')
        self._stop_logging()

        _logger.info('Experiment stopped')

    def export_top_models(self, top_k: int = 1, *,
                          formatter: Literal['code', 'dict', 'instance'] | None = None,
                          **kwargs) -> list[Any]:
        """Export several top performing models.

        The concrete behavior of export depends on each strategy.
        See the documentation of each strategy for detailed specifications.

        Parameters
        ----------
        top_k
            How many models are intended to be exported.
        formatter
            If formatter is none, original :class:`~nni.nas.space.ExecutableModelSpace` objects will be returned.
            Otherwise, the formatter will be used to convert the model space to a human-readable format.
            The formatter could be:

            - ``code``: the python code of model will be returned (only for :class:`~nni.nas.space.GraphModelSpace`).
            - ``dict``: the sample (architecture dict) that is used to freeze the model space.
            - ``instance``: the instantiated callable model.
        """

        if 'optimize_mode' in kwargs:
            warnings.warn('Optimize mode has no effect starting from NNI v3.0.', DeprecationWarning)

        models = list(self.strategy.list_models(limit=top_k))
        if formatter is None:
            return models
        if formatter == 'code':
            if not all(isinstance(model, GraphModelSpace) for model in models):
                raise ValueError('Formatter "code" is only supported for GraphModelSpace.')
            return [cast(GraphModelSpace, model).to_code() for model in models]
        if formatter == 'dict':
            return [model.sample for model in models]
        if formatter == 'instance':
            return [model.executable_model() for model in models]
        raise ValueError(f'Unsupported formatter: {formatter}')

    def _wait_completion(self) -> bool:
        _logger.info('Waiting for models submitted to engine to finish...')
        if self._engine is not None:
            self._engine.wait_models()
        _logger.info('Experiment is completed.')
        if self._nni_manager_required():
            _logger.info('Search process is done. You can put an `time.sleep(FOREVER)` '
                         'here to block the process if you want to continue viewing the experiment.')
        # Always return true no matter successful or not.
        return True

    def _nni_manager_required(self) -> bool:
        """Return whether NNI manager and training service are created.

        Use engine type (in config) as an indicator.
        """
        engine_config = self.config.canonical_copy().execution_engine
        return isinstance(engine_config, (TrainingServiceEngineConfig, CgoEngineConfig))

    def _send_space_to_nni_manager(self) -> None:
        """Make the search space informative on WebUI.

        Note: It doesn't work for complex search spaces (e.g., with mutators).
        """
        legacy_space_dict = {}
        for label, mutable in self.model_space.simplify().items():
            try:
                legacy_space_dict[label] = mutable.as_legacy_dict()
            except NotImplementedError:
                _logger.warning('Cannot convert %r to legacy format. It will not show on WebUI.', mutable)

        _logger.debug('Converted legacy space: %s', legacy_space_dict)
        self.update_search_space(legacy_space_dict)

    def load_checkpoint(self) -> None:
        """
        Recover the status of an experiment from checkpoint.

        It first loads the config, and then loads status for strategy and engine.
        The config must match exactly with the config used to create this experiment.

        The status of strategy and engine will only be loaded if engine has been created,
        and the checkpoint file exists.

        Notes
        -----
        This method is called twice when loading an experiment:

        - When resume is just called, the config will be loaded and will be cross-checked with the current config.
        - After NNI manager is started and engine is created, the full method is called to load the state of strategy and engine.
          For this time, the config will be loaded and cross-checked again.

        Semantically, "loading config" and "loading status" are two different things which should be done separately.
        The current implementation is a bit hacky, but it's simple and works.
        """

        # NAS experiment saves the config by itself. It doesn't rely on the HPO way which uses config from DB.
        with (self._checkpoint_directory / 'config.json').open('r') as f:
            config = NasExperimentConfig(**nni.load(fp=f))

        # NAS experiment MUST already have a config (because it will create one if not specified).
        from nni.experiment.config.utils import diff
        config_diff = diff(self.config, config, 'Current', 'Loaded')
        if config_diff:
            _logger.error('Config is found but does not match the current config:\n%s', config_diff)
            _logger.error(
                'NAS experiment loading will probably fail for inconsistent config. '
                'The experiment now continues with the new config.'
            )

        if self._engine is not None:
            ckpt_path = self._checkpoint_directory / 'state.ckpt'
            try:
                state_dict = get_default_serializer().load(ckpt_path)
                self.load_state_dict(state_dict)
                _logger.debug('State of engine and strategy is loaded from %s.', ckpt_path)
            except FileNotFoundError:
                _logger.warning('Checkpoint file %s does not exist. Skip loading.', ckpt_path)

    def save_checkpoint(self) -> None:
        """Save the whole experiment state.

        It will dump the config first (as a JSON) and then states of components like strategy and engine.
        It calls :meth:`state_dict` to get the states.
        """
        try:
            if not self._checkpoint_directory.exists():
                self._checkpoint_directory.mkdir(parents=True, exist_ok=True)

            with (self._checkpoint_directory / 'config.json').open('w', encoding='utf-8') as f:
                nni.dump(self.config.json(), fp=f, indent=2)

            if self._engine is not None:
                get_default_serializer().save(self.state_dict(), self._checkpoint_directory / 'state.ckpt')

            _logger.info('Checkpoint saved to %s.', self._checkpoint_directory)
        except:
            _logger.exception('Error occurred when saving checkpoint.')

    @property
    def _checkpoint_directory(self) -> Path:
        """The checkpoint directory of a NAS experiment. Considered internal for now.

        It could change depending on the contents in ``self.config``.
        """
        return Path(self.config.canonical_copy().experiment_working_directory) / self.id / 'checkpoint'

    def state_dict(self) -> dict:
        """Summarize the state of current experiment for serialization purposes.

        Please deepcopy the states (or save them to disk) in case you want to restore them later.

        NOTE: This should only be called after the engine is created (i.e., after calling :meth:`start`).
        """
        result = {
            'version': self._state_dict_version,
            'strategy': self.strategy.state_dict(),
        }
        if self._engine is not None:
            result['engine'] = self._engine.state_dict()
        return result

    def load_state_dict(self, state_dict: dict):
        """Load the state dict to recover the status of experiment.

        NOTE: This should only be called after the engine is created (i.e., after calling :meth:`start`).
        """
        if state_dict['version'] != self._state_dict_version:
            _logger.warning(f'Incompatible state dict version: {state_dict["version"]} vs {self._state_dict_version}. '
                            'Some components may not be restored correctly.')
        if self._engine is not None:
            self._engine.load_state_dict(state_dict['engine'])
        self.strategy.load_state_dict(state_dict['strategy'])
