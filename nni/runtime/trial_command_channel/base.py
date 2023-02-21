# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing_extensions import Literal

from nni.typehint import ParameterRecord, TrialMetric

class TrialCommandChannel:
    """
    Command channel used by trials to communicate with training service.

    One side of this channel is trial, who asks for parameters and reports metrics.
    The other side of this channel is training service, which launches and manages trial jobs.

    Due to the complexity of training environments,
    :class:`TrialCommandChannel` might have multiple implementations.
    The underlying implementation of :class:`TrialCommandChannel` usually
    relies on network communication, shared file system, etc,
    which is covered in :class:`~nni.runtime.command_channel.base.CommandChannel`.
    """

    def receive_parameter(self) -> ParameterRecord | None:
        """Get the next parameter record from NNI manager.

        Returns
        -------
        :class:`~nni.typehint.ParameterRecord`
            The next parameter record.
            Could be ``None`` if no more parameter is available.
        """
        raise NotImplementedError()

    def send_metric(
        self,
        type: Literal['PERIODICAL', 'FINAL'],  # pylint: disable=redefined-builtin
        parameter_id: int | None,
        trial_job_id: str,
        sequence: int,
        value: TrialMetric,
    ) -> None:
        """Send a metric to NNI manager.

        Parameters
        ----------
        type
            Type of the metric. Must be ``'PERIODICAL'`` or ``'FINAL'``.
        parameter_id
            ID of the parameter. Could be ``None`` if no parameter is associated with the metric.
        trial_job_id
            ID of the trial job.
        sequence
            Sequence number of the metric. Only meaningful for intermediate metrics.
            Must be ``0`` for final metrics.
        value
            The metric value.
        """
        raise NotImplementedError()
