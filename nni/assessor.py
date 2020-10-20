# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Assessor analyzes trial's intermediate results (e.g., periodically evaluated accuracy on test dataset)
to tell whether this trial can be early stopped or not.

See :class:`Assessor`' specification and ``docs/en_US/assessors.rst`` for details.
"""

from enum import Enum
import logging

from .recoverable import Recoverable

__all__ = ['AssessResult', 'Assessor']

_logger = logging.getLogger(__name__)


class AssessResult(Enum):
    """
    Enum class for :meth:`Assessor.assess_trial` return value.
    """

    Good = True
    """The trial works well."""

    Bad = False
    """The trial works poorly and should be early stopped."""


class Assessor(Recoverable):
    """
    Assessor analyzes trial's intermediate results (e.g., periodically evaluated accuracy on test dataset)
    to tell whether this trial can be early stopped or not.

    This is the abstract base class for all assessors.
    Early stopping algorithms should inherit this class and override :meth:`assess_trial` method,
    which receives intermediate results from trials and give an assessing result.

    If :meth:`assess_trial` returns :obj:`AssessResult.Bad` for a trial,
    it hints NNI framework that the trial is likely to result in a poor final accuracy,
    and therefore should be killed to save resource.

    If an accessor want's to be notified when a trial ends, it can also override :meth:`trial_end`.

    To write a new assessor, you can reference :class:`~nni.medianstop_assessor.MedianstopAssessor`'s code as an example.

    See Also
    --------
    Builtin assessors:
    :class:`~nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor`
    :class:`~nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor`
    """

    def assess_trial(self, trial_job_id, trial_history):
        """
        Abstract method for determining whether a trial should be killed. Must override.

        The NNI framework has little guarantee on ``trial_history``.
        This method is not guaranteed to be invoked for each time ``trial_history`` get updated.
        It is also possible that a trial's history keeps updating after receiving a bad result.
        And if the trial failed and retried, ``trial_history`` may be inconsistent with its previous value.

        The only guarantee is that ``trial_history`` is always growing.
        It will not be empty and will always be longer than previous value.

        This is an example of how :meth:`assess_trial` get invoked sequentially:

        ::

            trial_job_id | trial_history   | return value
            ------------ | --------------- | ------------
            Trial_A      | [1.0, 2.0]      | Good
            Trial_B      | [1.5, 1.3]      | Bad
            Trial_B      | [1.5, 1.3, 1.9] | Good
            Trial_A      | [0.9, 1.8, 2.3] | Good

        Parameters
        ----------
        trial_job_id : str
            Unique identifier of the trial.
        trial_history : list
            Intermediate results of this trial. The element type is decided by trial code.

        Returns
        -------
        AssessResult
            :obj:`AssessResult.Good` or :obj:`AssessResult.Bad`.
        """
        raise NotImplementedError('Assessor: assess_trial not implemented')

    def trial_end(self, trial_job_id, success):
        """
        Abstract method invoked when a trial is completed or terminated. Do nothing by default.

        Parameters
        ----------
        trial_job_id : str
            Unique identifier of the trial.
        success : bool
            True if the trial successfully completed; False if failed or terminated.
        """

    def load_checkpoint(self):
        """
        Internal API under revising, not recommended for end users.
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Load checkpoint ignored by assessor, checkpoint path: %s', checkpoin_path)

    def save_checkpoint(self):
        """
        Internal API under revising, not recommended for end users.
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Save checkpoint ignored by assessor, checkpoint path: %s', checkpoin_path)

    def _on_exit(self):
        pass

    def _on_error(self):
        pass
