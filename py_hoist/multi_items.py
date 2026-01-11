from collections.abc import Iterable
from typing import Generic, TypeVar, Iterator
from dataclasses import dataclass

K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots = True)
class MultiItems(Generic[K, V]):
    _items: list[tuple[K, V]] = None
    _keys: list[K] = None
    _values: list[V] = None

    def __init__(self, items: Iterable[tuple[K, V]] = ()):
        self._items = list(items)
        self._keys = None
        self._values = None

    def add(self, key: K, value: V) -> None:
        self._items.append((key, value))
        self._keys = None
        self._values = None

    # ---- dict 风格接口 ----
    def keys(self) -> list[K]:
        _keys = self._keys
        if _keys is None:
            _keys = [k for k, _ in self._items]
            self._keys = _keys
        return _keys

    def values(self) -> list[V]:
        _values = self._values
        if _values is None:
            _values = [v for _, v in self._items]
            self._values = _values
        return _values

    def items(self) -> list[tuple[K, V]]:
        return self._items

    def __iter__(self):
        return self.items()

    def __len__(self):
        return len(self._items)
