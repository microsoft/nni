# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""This module does what
`CheckpointIO <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.plugins.io.CheckpointIO.html>`__
does in PyTorch-Lightning, but with a simpler implementation and a wider range of backend supports.

The default implementation is :class:`TorchSerializer`, which uses :func:`torch.save` and :func:`torch.load`
to save and load the data. But it can't be used in some cases, e.g., when torch is not installed,
or when the data requires special handling that are not supported by torch.

There are several alternatives, which can be switched via :func:`set_default_serializer`.
The serializer defined in :mod:`nni.common.serializer` happened to be one of the alternatives.

NOTE: The file is placed in NAS experimentally. It might be merged with the global serializer in near future.
"""

from __future__ import annotations

__all__ = ['Serializer', 'set_default_serializer', 'get_default_serializer', 'TorchSerializer', 'JsonSerializer']

import logging
from pathlib import Path
from typing import Any, ClassVar, Type, cast

_logger = logging.getLogger(__name__)


class Serializer:
    """Save data to a file, or load data from a file."""

    suffix: ClassVar[str] = ''
    """All serializers should save the file with a suffix,
    which is used to validate the serializer type when loading data.
    """

    def save(self, data: Any, path: Path):
        """Save the data to a given path. The path might be suffixed with :attr:`suffix`."""
        raise NotImplementedError()

    def load(self, path: Path) -> Any:
        """Load the data from the given path.

        Raises
        ------
        FileNotFoundError
            If the file (suffixed with :attr:`suffix`) is not found.

        """
        raise NotImplementedError()


_default_serializer: Serializer | None = None


def set_default_serializer(serializer: Serializer) -> None:
    """Set the default serializer.

    Parameters
    ----------
    serializer
        The serializer to be used as default.
    """
    global _default_serializer
    _default_serializer = serializer


def get_default_serializer() -> Serializer:
    """Get the default serializer.

    Returns
    -------
    The default serializer.
    """
    if _default_serializer is None:
        set_default_serializer(TorchSerializer())
    return cast(Serializer, _default_serializer)


def _find_path_with_prefix(path: Path, expected_suffix: str) -> Path:
    """Find a file that is prefixed with the given path.

    Parameters
    ----------
    path
        The path to be searched.
    expected_suffix
        The suffix of the file we want.
    """
    if path.exists():
        return path

    for p in path.parent.iterdir():
        if p.name.startswith(path.name) and p.name == path.name + expected_suffix:
            return p

    # Iter again to give a warning.
    for p in path.parent.iterdir():
        if p.name.startswith(path.name):
            # Try to find the serializer type that can load this file.
            guessed_serializer_type: Type[Serializer] | None = None
            for serializer in Serializer.__subclasses__():
                if p.name == path.name + serializer.suffix:
                    guessed_serializer_type = serializer
            if guessed_serializer_type is not None:
                _logger.warning('Found file %s, which could be loaded by %s, but suffix %s is expected.',
                                p, guessed_serializer_type, expected_suffix)
            else:
                _logger.warning('Found file %s, which does not match any serializer registered. Suffix %s is expected.',
                                p, expected_suffix)

    raise FileNotFoundError(f'No file found with prefix {path} and suffix {expected_suffix}.')


class TorchSerializer(Serializer):
    """The serializer that utilizes :func:`torch.save` and :func:`torch.load` to save and load data.

    This serializer should work in most scenarios, including cases when strategies have some tensors in their states (e.g., DRL).
    The downside is that it relies on torch to be installed.

    Parameters
    ----------
    map_location
        The ``map_location`` argument to be passed to :func:`torch.load`.
    """

    suffix: ClassVar[str] = '.torch'

    def __init__(self, map_location: Any = None):
        try:
            import torch  # pylint: disable=unused-import
        except ImportError:
            raise RuntimeError(
                'TorchSerializer requires torch to be installed. '
                'Either install torch or set a different serializer. '
                'For example, `nni.nas.serializer.set_default_serializer(nni.nas.serializer.JsonSerializer())`.'
            )

        self._map_location = map_location

    def save(self, checkpoint: Any, path: Path):
        import torch
        torch.save(checkpoint, str(path) + self.suffix)

    def load(self, path: Path):
        path = _find_path_with_prefix(path, self.suffix)
        import torch
        return torch.load(str(path), map_location=self._map_location)


class JsonSerializer(Serializer):
    """The serializer that utilizes :func:`nni.dump` and :func:`nni.load` to save and load data.

    This serializer should work in cases where strategies have no complex objects in their states.
    Since it uses :func:`nni.dump`, it resorts to binary format when some part of the data is not JSON-serializable.
    """

    suffix: ClassVar[str] = '.json'

    def save(self, checkpoint: Any, path: Path):
        import nni
        with (path.parent / (path.name + self.suffix)).open('w') as f:
            nni.dump(checkpoint, fp=f, indent=2)

    def load(self, path: Path):
        import nni
        path = _find_path_with_prefix(path, self.suffix)
        with path.open() as f:
            return nni.load(fp=f)
