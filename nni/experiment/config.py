# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Experiment configuration structures.
"""

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

__all__ = [
    'LocalExperimentConfig',
    #'RemoteExperimentConfig',
    #'OpenPaiExperimentConfig',
]

_TrainingServiceEnum = Literal['local', 'remote', 'pai']


@dataclass
class ExperimentConfig:
    """
    Base class of experiment configuration.

    End user should use one of its derived classes instead:

    * `LocalExperimentConfig`
    * `RemoteExperimentConfig`
    * `OpenPaiExperimentConfig`
    """

    trial_concurrency: int
    """
    Specify how many trials could be run concurrently.
    """

    search_space: Any  # TODO: consider move into tuner
    """
    Search space of hyper-parameters.

    The common format for built-in tuners is specified in `search space doc`_.

    .. _search space doc: https://nni.readthedocs.io/en/stable/Tutorial/SearchSpaceSpec.html
    """

    trial_command: str  # TODO: alternatively specify a Python function or module
    """
    Command line to launch trial process, for example `python3 trial.py`.

    Working directory of this command is assumed to be `trial_code_directory`.
    """

    trial_code_directory: Union[Path, str]
    """
    The directory which contains all trial code, and maybe data.

    It is recommended to use a path relative to caller's script file,
    for example `Path(__file__).parent`.
    """

    _training_service: _TrainingServiceEnum

    experiment_name: Optional[str] = None
    """
    Mnemonic description of the experiment.
    """

    max_execution_seconds: Optional[int] = None
    """
    If specified, the experiment will stop after this number of seconds.
    """

    max_trial_number: Optional[int] = None
    """
    If specified, the experiment will stop after this number of trials complete.
    """

    trial_gpu_number: int = 0
    """
    Specify number of GPUs used by (and visible to) each trial.
    """

    extra_config: Optional[Dict[str, str]] = None
    """
    Key-value pairs for less commonly used configuration.

    This can be safely ignored for most users.
    """

    def validate(self) -> None:
        """
        Validate the fields and raise `ValueError` or `TypeError` if anything wrong.

        This will not check search space.
        """
        for key, placeholder_value in type(self)._placeholders.items():
            if getattr(self, key) == placeholder_value:
                raise ValueError(f'Field "{key}" is not set')

        if not isinstance(self.trial_concurrency, int):
            raise TypeError('Field "trial_concurrency" is not int')
        if self.trial_concurrency <= 0:
            raise ValueError('Field "trial_concurrency" must greater than zero"')

        # TODO


    @staticmethod
    def _create(training_service: str) -> 'ExperimentConfig':
        # create an empty configuration for given training service
        for cls in ExperimentConfig.__subclasses__():
            for field in fields(cls):
                if field.name == '_training_service' and field.default == training_service:
                    return cls(**cls._placeholders)
        raise ValueError(f'Unrecognized training service {training_service}')

    def _to_json(self) -> Dict[str, Any]:  # TODO: make this more elegant
        ret = {}
        ret['authorName'] = ''
        ret['experimentName'] = self.experiment_name
        ret['trialConcurrency'] = self.trial_concurrency
        ret['masExecDuration'] = str(self.max_execution_seconds) + 's'
        ret['maxTrialNum'] = self.max_trial_num
        ret['searchSpace'] = json.dumps(self.search_space)
        ret['trainingServicePlatform'] = self._training_service
        ret['tuner'] = {'builtinTunerName': '_placeholder_'}
        ret['clusterMetaData'] = []
        if self.extra_config:
            ret.update(self.extra_config)
        return ret

    _placeholders = {
        'trial_concurrency': -1,
        'search_space': '_unset_',
        'trial_command': '_unset_',
        'trial_code_directory': '_unset_'
    }


@dataclass
class LocalExperimentConfig(ExperimentConfig):
    """
    Experiment configuration for "local" training service.
    """

    use_active_gpu: bool = False  # TODO: consider change default value to true, "local" machine is less commonly shared
    """
    By default NNI will not use GPUs which already have running tasks.

    Set this field to `True` to change this behavior.

    For example if the machine is running a GUI this should be set to True;
    and if the machine is a server shared by a team it should be set to False.
    """

    _training_service: _TrainingServiceEnum = 'local'

    def _to_json(self) -> Dict[str, Any]:
        ret = super()._to_json()
        ret['clusterMetaData']['codeDir'] = str(self.trial_code_directory)
        ret['clusterMetaData']['command'] = self.trial_command
        return ret


#@dataclass
#class RemoteExperimentConfig(ExperimentConfig):
#    """
#    Experiment configuration for "remote" training service.
#    """
#
#    machines: List['RemoteMachineConfig']
#    """
#    Training machines' information. See `RemoteMachineConfig` for details.
#    """
#
#    use_active_gpu: bool = False
#    """
#    By default NNI will not use GPUs which already have running tasks.
#
#    Set this field to `True` to change this behavior.
#
#    For example if the machine is running a GUI this should be set to True;
#    and if the machine is a server shared by a team it should be set to False.
#    """
#
#    #trial_prepare_command: Optional[str] = None
#    # FIXME: what's the actual difference to trial command? or if it's merely for elegant?
#
#    _training_service: _TrainingServiceEnum = 'remote'
#
#    _placeholders = {
#        **ExperimentConfig._placeholders,
#        'machines': []
#    }
#
#    def _to_json(self) -> Dict[str, Any]:
#        ret = super()._to_json()
#        ret['clusterMetaData']['machineList'] = []
#        ret['clusterMetaData']['trialConfig'] = {
#            'codeDir': str(self.trial_code_directory),
#            'command': self.trial_command
#        }
#        ret['clusterMetaData']['remoteConfig'] = {'reuse': False}
#        for machine in self.machines:
#            ret['clusterMetaData']['machineList'].append({
#                'ip': machine.ip,
#                'port': machine.port,
#                'username': machine.user_name,
#                'passwd': machine.password,
#                'sshKeyPath': machine.ssh_key_path,
#                'passphrase': machine.ssh_key_pass_phrase,
#                'useActiveGpu': self.use_active_gpu
#            })
#        return ret
#
#
#@dataclass
#class RemoteMachineConfig(NamedTuple):
#    """
#    Configuration of each training machine for "remote" training service.
#    """
#
#    ip: str
#    """
#    IP address or domain name.
#    """
#
#    port: int = 22
#    """
#    SSH service port.
#    """
#
#    user_name: str
#    """
#    Login name.
#    """
#
#    password: Optional[str] = None
#    """
#    Login password.
#
#    If this is not set, SSH key will be used instead.
#    """
#
#    ssh_key_path: Optional[Path] = Path.home() / '.ssh/id_rsa'
#    """
#    File path to SSH private key.
#
#    This is only used when `password` is not set.
#    """
#
#    ssh_key_pass_phrase: Optional[str]
#    """
#    Pass phrase of SSH key.
#    """
#
#
#@dataclass
#class OpenPaiExperimentConfig(ExperimentConfig):
#    """
#    Experiment configuration for OpenPAI training service.
#    """
#
#    openpai_host: str
#    """
#    URL of OpenPAI cluster.
#
#    For example `https://mycluster.openpai.org`.
#    """
#
#    openpai_user_name: str
#    """
#    OpenPAI User name.
#    """
#
#    openpai_token: str
#    """
#    OpenPAI access token.
#
#    This can be found in your user settings page on OpenPAI.
#    """
#
#    _training_service: _TrainingServiceEnum = 'pai'
#
#    _placeholders = {
#        **ExperimentConfig._placeholders,
#        'openpai_host': '_unset_',
#        'openpai_user_name': '_unset_',
#        'openpai_token': '_unset_'
#    }
#
#    def _to_json(self) -> Dict[str, Any]:
#        ret = super()._to_json()
#        ret['clusterMetaData']['pai_config'] = {
#            'userName': self.openpai_user_name,
#            'token': self.openpai_token,
#            'host': self.openpai_host,
#            'reuse': False
#        }
#        ret['clusterMetaData']['trial_config'] = {
#            'codeDir': str(self.trial_code_directory),
#            'command': self.trial_command
#        }
#        return ret
