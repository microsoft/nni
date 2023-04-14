# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['Metrics']

from typing import Any, Sequence, cast

from nni.typehint import TrialMetric


class Metrics:
    """
    Data structure that manages the metric data (e.g., loss, accuracy, etc.).

    NOTE: Multiple metrics and minimized metrics are not supported in the current iteration.

    Parameters
    ----------
    strict
        Whether to convert the metrics into a float.
        If ``true``, only float metrics or dict with "default" are accepted.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict

        self._intermediates: list[TrialMetric] = []
        self._final: TrialMetric | None = None

    def __bool__(self):
        """Return whether the metrics has been (at least partially) filled."""
        return bool(self.intermediates or self.final)

    def __repr__(self):
        return f"Metrics(intermediates=<array of length {len(self.intermediates)}>, final={self.final})"

    def _dump(self) -> dict:
        rv: dict[str, Any] = {'intermediates': self._intermediates}
        if self.final is not None:
            rv['final'] = self.final
        return rv

    @classmethod
    def _load(cls, intermediates: list[TrialMetric], final: TrialMetric | None = None) -> Metrics:
        rv = Metrics()
        rv._intermediates = intermediates
        rv._final = final
        return rv

    def add_intermediate(self, metric: TrialMetric) -> None:
        self._intermediates.append(self._canonicalize_metric(metric))

    def clear(self) -> None:
        self._intermediates.clear()
        self._final = None

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Metrics) and self._intermediates == other._intermediates and self._final == other._final

    @property
    def intermediates(self) -> Sequence[TrialMetric]:
        return self._intermediates

    @property
    def final(self) -> TrialMetric | None:
        return self._final

    @final.setter
    def final(self, metric: TrialMetric) -> None:
        self._final = self._canonicalize_metric(metric)

    def _canonicalize_metric(self, metric: Any) -> TrialMetric:
        if not self.strict:
            return cast(TrialMetric, metric)
        if isinstance(metric, dict):
            if 'default' not in metric:
                raise ValueError(f"Metric dict {metric} does not contain key 'default'")
            metric = metric['default']
        if not isinstance(metric, (int, float)):
            raise ValueError(f"Metric {metric} is not a number")
        return float(metric)
