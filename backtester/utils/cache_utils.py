"""Caching utilities for performance optimization."""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any

import pandas as pd


class CacheUtils:
    """Utility class for caching data in memory and file system."""

    def __init__(
        self,
        memory_cache: bool = True,
        cache_dir: str | None = None,
        max_memory_items: int = 1000,
    ):
        """Initialize cache utilities.

        Args:
            memory_cache: Enable memory caching
            cache_dir: Directory for file caching (None to use temp directory)
            max_memory_items: Maximum number of items to keep in memory cache
        """
        self.memory_cache = memory_cache
        self.cache_dir: str | None = cache_dir or tempfile.gettempdir()
        self.max_memory_items = max_memory_items

        # Memory cache storage
        self._memory_cache: dict[str, Any] = {}
        self._memory_cache_times: dict[str, float | None] = {}
        self._cache_lock = threading.RLock()

        # Statistics
        self._hits: int = 0
        self._misses: int = 0

        # Ensure cache directory exists
        if self.cache_dir:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except OSError:
                # Fallback to in-memory cache only when file system operations fail.
                self.cache_dir = None

    def _get_cache_key(self, key: str) -> str:
        """Generate cache key from input key."""
        return hashlib.md5(str(key).encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        """Get file path for cache key."""
        if self.cache_dir is None:
            msg = "File-based caching is disabled; no cache directory available"
            raise RuntimeError(msg)
        return os.path.join(self.cache_dir, f"cache_{cache_key}.pkl")

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiry)
        """
        cache_key = self._get_cache_key(key)
        expiry_time = time.time() + ttl if ttl else None

        with self._cache_lock:
            # Memory cache
            if self.memory_cache:
                self._memory_cache[cache_key] = value
                self._memory_cache_times[cache_key] = expiry_time

                # Limit memory cache size
                if len(self._memory_cache) > self.max_memory_items:
                    self._cleanup_memory_cache()

            # File cache
            if self.cache_dir:
                cache_data = {
                    'value': value,
                    'expiry_time': expiry_time,
                    'created_time': time.time(),
                }

                try:
                    file_path = self._get_cache_file_path(cache_key)
                    with open(file_path, 'wb') as f:
                        pickle.dump(cache_data, f)
                except Exception:
                    pass  # Ignore file cache errors

    def get(self, key: str, force_check: bool = False) -> Any | None:
        """Get a value from cache.

        Args:
            key: Cache key
            force_check: Force recheck of expiry times

        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._get_cache_key(key)
        current_time = time.time()

        with self._cache_lock:
            # Check memory cache first
            if self.memory_cache and cache_key in self._memory_cache:
                expiry_time = self._memory_cache_times.get(cache_key)

                # Treat None (no expiry) as always valid
                if expiry_time is None or current_time < expiry_time:
                    self._hits += 1
                    return self._memory_cache[cache_key]
                else:
                    # Expired, remove from memory cache
                    self._remove_from_memory_cache(cache_key)

            # Check file cache if enabled
            if self.cache_dir:
                try:
                    file_path = self._get_cache_file_path(cache_key)
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            cache_data = pickle.load(f)

                        expiry_time = cache_data.get('expiry_time')

                        if not expiry_time or current_time < expiry_time:
                            self._hits += 1

                            # Update memory cache if enabled
                            if self.memory_cache and cache_key not in self._memory_cache:
                                self._memory_cache[cache_key] = cache_data['value']
                                self._memory_cache_times[cache_key] = expiry_time

                            return cache_data['value']
                        else:
                            # Expired, remove file
                            os.remove(file_path)
                except Exception:
                    pass  # Ignore file cache errors

            self._misses += 1
            return None

    def invalidate(self, key: str) -> None:
        """Invalidate a specific cache key.

        Args:
            key: Cache key to invalidate
        """
        cache_key = self._get_cache_key(key)

        with self._cache_lock:
            # Remove from memory cache
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
                del self._memory_cache_times[cache_key]

            # Remove from file cache
            if self.cache_dir:
                try:
                    file_path = self._get_cache_file_path(cache_key)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass  # Ignore file cache errors

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock:
            # Clear memory cache
            self._memory_cache.clear()
            self._memory_cache_times.clear()

            # Clear file cache
            if self.cache_dir:
                try:
                    for filename in os.listdir(self.cache_dir):
                        if filename.startswith("cache_") and filename.endswith(".pkl"):
                            file_path = os.path.join(self.cache_dir, filename)
                            with contextlib.suppress(Exception):
                                os.remove(file_path)
                except Exception:
                    pass  # Ignore directory access errors

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            'hits': float(self._hits),
            'misses': float(self._misses),
            'hit_rate': float(hit_rate),
            'total_requests': float(total_requests),
            'memory_cache_size': float(len(self._memory_cache)),
            'max_memory_cache_size': float(self.max_memory_items),
        }

    def _cleanup_memory_cache(self) -> None:
        """Clean up memory cache when it exceeds maximum size."""
        # Sort by access time (oldest first) and remove excess items
        cache_items = [
            (key, self._memory_cache_times.get(key) or 0.0) for key in self._memory_cache
        ]
        cache_items.sort(key=lambda x: x[1])

        # Remove oldest items
        items_to_remove = len(self._memory_cache) - self.max_memory_items
        for i in range(items_to_remove):
            key = cache_items[i][0]
            del self._memory_cache[key]
            del self._memory_cache_times[key]

    def _remove_from_memory_cache(self, cache_key: str) -> None:
        """Remove item from memory cache."""
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        if cache_key in self._memory_cache_times:
            del self._memory_cache_times[cache_key]


class FrameCache:
    """Thread-safe in-memory cache optimized for pandas DataFrames."""

    def __init__(self, max_entries: int = 256) -> None:
        """Create a cache with the specified capacity."""
        self._lock = threading.RLock()
        self._frames: OrderedDict[Hashable, tuple[pd.DataFrame, float | None]] = OrderedDict()
        self._max_entries = max(1, int(max_entries))

    def configure(self, *, max_entries: int | None = None) -> None:
        """Update cache capacity limit."""
        if max_entries is None:
            return
        with self._lock:
            self._max_entries = max(1, int(max_entries))
            self._evict_if_needed()

    def get(self, key: Hashable) -> pd.DataFrame | None:
        """Retrieve a cached frame copy."""
        with self._lock:
            entry = self._frames.get(key)
            if entry is None:
                return None
            frame, expire_at = entry
            if expire_at is not None and time.time() >= expire_at:
                del self._frames[key]
                return None
            self._frames.move_to_end(key)
            return frame.copy(deep=True)

    def set(self, key: Hashable, frame: pd.DataFrame, ttl: float | None = None) -> None:
        """Store a frame in the cache."""
        expire_at = time.time() + ttl if ttl else None
        with self._lock:
            self._frames[key] = (frame.copy(deep=True), expire_at)
            self._frames.move_to_end(key)
            self._evict_if_needed()

    def invalidate(self, key: Hashable) -> None:
        """Remove a cached entry."""
        with self._lock:
            self._frames.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._frames.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics for diagnostics."""
        with self._lock:
            expirations = [expiry for _, (_, expiry) in self._frames.items()]
            return {
                'size': len(self._frames),
                'max_entries': self._max_entries,
                'expiring_entries': sum(1 for value in expirations if value is not None),
            }

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _evict_if_needed(self) -> None:
        while len(self._frames) > self._max_entries:
            self._frames.popitem(last=False)
