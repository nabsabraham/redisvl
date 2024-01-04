import pickle
import zlib
import redis

from typing import List, Optional, Dict, Any



class EmbeddingCache(BaseLLMCache):
    """
    A cache class for storing and retrieving vector embeddings using Redis.
    This class extends the BaseLLMCache for specific use with vector embeddings.
    """
    _vector_key: str = "vector"
    _metadata_key: str = "metadata"

    def __init__(self, prefix: str, redis_client: redis.Redis, ttl: Optional[int] = None):
        """
        Initialize the EmbeddingCache with a Redis client, key prefix, and optional TTL.
        Args:
            redis_client (redis.Redis): The Redis client for cache operations.
            prefix (str): Prefix for the cache keys.
            ttl (Optional[int], optional): Time-to-live for the cache entries.
        """
        super().__init__(prefix, redis_client, ttl)

    def _prefixed_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    @staticmethod
    def _compress_value(data: Dict[str, Any]) -> bytes:
        return zlib.compress(pickle.dumps(data))

    @staticmethod
    def _decompress_value(data: bytes) -> Dict[str, Any]:
        return pickle.loads(zlib.decompress(data)) if data else None

    def _prepare_store_value(self, entry: Dict[str, Any]) -> bytes:
        return self._compress_value({
            self._vector_key: entry['vector'],
            self._metadata_key: entry.get('metadata', {})
        })

    def store(self, entries: List[Dict[str, Any]], batch: bool = False) -> None:
        """
        Stores single or multiple entries in the cache.
        Args:
            entries (List[Dict[str, Any]]): List containing 'prompt', 'vector', and 'metadata'.
            batch (bool): If True, uses batch processing.
        """
        if batch:
            self._store_batch(entries)
        else:
            for entry in entries:
                self._store_single(entry)

    def _store_single(self, entry: Dict[str, Any]) -> None:
        key = self._prefixed_key(self.hash_input(entry['prompt']))
        value = self._prepare_store_value(entry)
        self.redis_client.set(key, value, ex=self._ttl)

    def _store_batch(self, entries: List[Dict[str, Any]]) -> None:
        with self.redis_client.pipeline(transaction=False) as pipe:
            for entry in entries:
                key = self._prefixed_key(self.hash_input(entry['prompt']))
                value = self._prepare_store_value(entry)
                pipe.set(key, value, ex=self._ttl)
            pipe.execute()

    def check(self, prompts: List[str], batch: bool = False) -> List[Optional[Dict[str, Any]]]:
        """
        Checks single or multiple prompts in the cache.
        Args:
            prompts (List[str]): Prompts to check in the cache.
            batch (bool): If True, uses batch processing.
        """
        if batch:
            return self._check_batch(prompts)
        else:
            return [self._check_single(prompt) for prompt in prompts]

    def _check_single(self, prompt: str) -> Optional[Dict[str, Any]]:
        key = self._prefixed_key(self.hash_input(prompt))
        result = self.redis_client.get(key)
        return self._decompress_value(result) if result else None

    def _check_batch(self, prompts: List[str]) -> List[Optional[Dict[str, Any]]]:
        with self.redis_client.pipeline(transaction=False) as pipe:
            for prompt in prompts:
                key = self._prefixed_key(self.hash_input(prompt))
                pipe.get(key)
            results = pipe.execute()
        return [self._decompress_value(result) for result in results]

    def clear(self) -> None:
        """
        Clears all entries with the specified prefix in the cache.
        """
        try:
            with self.redis_client.pipeline(transaction=False) as pipe:
                for key in self.redis_client.scan_iter(f"{self.prefix}:*"):
                    pipe.delete(key)
                pipe.execute()
        except Exception as e:
            raise RuntimeError(f"Error clearing cache: {e}")
