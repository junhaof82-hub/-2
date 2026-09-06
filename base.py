from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import pandas as pd


class DataProviderError(RuntimeError):
    pass


@dataclass
class CacheItem:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, ttl_seconds: int = 900):
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, CacheItem] = {}

    def get(self, key: str):
        item = self._data.get(key)
        if not item or item.expires_at < time.time():
            self._data.pop(key, None)
            return None
        value = item.value
        return value.copy() if hasattr(value, 'copy') else value

    def set(self, key: str, value: Any):
        self._data[key] = CacheItem(value=value.copy() if hasattr(value, 'copy') else value,
                                    expires_at=time.time() + self.ttl_seconds)


class MarketDataProvider(ABC):
    name = 'base'

    @abstractmethod
    def history(self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        raise NotImplementedError
