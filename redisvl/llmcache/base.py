import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Any

from redis import Redis


class BaseLLMCache(ABC):

    def __init__(self, prefix: str, redis_client: Redis, ttl: Optional[int] = None):
        self.prefix = prefix
        self.redis_client = redis_client
        self._ttl = ttl

    @property
    def ttl(self) -> Optional[int]:
        return self._ttl

    def set_ttl(self, ttl: Optional[int] = None):
        if ttl and not isinstance(ttl, int):
            raise ValueError(f"TTL must be an integer value, got {ttl}")
        self._ttl = ttl

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def check(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def store(self, *args, **kwargs) -> None:
        pass

    def hash_input(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
