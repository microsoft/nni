# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['Metrics']

from nni.typehint import TrialMetric

class Metrics:
    """
    Data structure that manages the metric data (e.g., loss, accuracy, etc.).
    """

    # TODO: Features related to metrics need more designs: metric formats, intermediates, multiple metrics, etc.

    def __init__(self):
        self.intermediates: list[TrialMetric] = []
        self.final: TrialMetric | None = None

    def __bool__(self):
        """Return whether the metrics has been (at least partially) filled."""
        return bool(self.intermediates or self.final)

    def __repr__(self):
        return f"Metrics(intermediates=<array of length {len(self.intermediates)}>, final={self.final})"

    def _dump(self) -> dict:
        rv = {'intermediates': self.intermediates}
        if self.final is not None:
            rv['final'] = self.final
        return rv

    @classmethod
    def _load(cls, intermediates: list[TrialMetric], final: TrialMetric | None = None) -> Metrics:
        rv = Metrics()
        rv.intermediates = intermediates
        rv.final = final
        return rv
