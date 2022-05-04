from abc import abstractmethod
from typing import Protocol, TypeVar


__all__ = ['ComparableType']

class Comparable(Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self, other) -> bool:
        pass

ComparableType = TypeVar("ComparableType", bound=Comparable)
