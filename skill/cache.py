"""Small in-memory TTL cache used by the QA runtime."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass
class _Entry(Generic[V]):
    value: V
    expires_at: float


class TTLCache(Generic[K, V]):
    """Bounded process-local cache with second-level TTL expiry."""

    def __init__(self, maxsize: int = 128, ttl_seconds: int = 300) -> None:
        self._maxsize = max(1, maxsize)
        self._ttl_seconds = max(1, ttl_seconds)
        self._items: OrderedDict[K, _Entry[V]] = OrderedDict()

    def get(self, key: K) -> V | None:
        entry = self._items.get(key)
        if entry is None:
            return None

        now = time.time()
        if entry.expires_at <= now:
            self._items.pop(key, None)
            return None

        self._items.move_to_end(key)
        return entry.value

    def set(self, key: K, value: V) -> None:
        self._items[key] = _Entry(value=value, expires_at=time.time() + self._ttl_seconds)
        self._items.move_to_end(key)
        while len(self._items) > self._maxsize:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()
