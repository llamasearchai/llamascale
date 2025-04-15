#!/usr/bin/env python3
"""
Hybrid Cache System - Multi-level caching with memory, disk, and Redis
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class HybridCache:
    """Multi-level cache with semantic similarity matching"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize cache with configuration

        Args:
            config: Cache configuration with the following options:
                - memory_size: Maximum number of items in memory cache
                - disk_path: Path to disk cache
                - redis_url: Redis connection URL (optional)
                - ttl: Default TTL in seconds
                - semantic_threshold: Threshold for semantic similarity matching
        """
        self.config = config
        self.memory = {}
        self.memory_lru = []  # LRU tracking
        self.memory_size = config.get("memory_size", 1000)

        # Initialize Redis if configured
        redis_url = config.get("redis_url")
        self.redis = None
        if redis_url:
            try:
                import redis.asyncio as redis

                self.redis = redis.Redis.from_url(redis_url)
                logger.info(f"Redis cache initialized with URL: {redis_url}")
            except ImportError:
                logger.warning("Redis package not available, Redis caching disabled")

        # Initialize disk cache
        disk_path = config.get("disk_path")
        self.disk_path = disk_path
        if disk_path:
            os.makedirs(disk_path, exist_ok=True)
            logger.info(f"Disk cache initialized at: {disk_path}")

        # Default TTL
        self.ttl = config.get("ttl", 300)  # 5 minutes default

        # Statistics
        self.stats = {
            "memory_hits": 0,
            "redis_hits": 0,
            "disk_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "sets": 0,
        }

    def _hash_key(self, key: str) -> str:
        """Hash a key string for storage"""
        return hashlib.md5(key.encode()).hexdigest()

    def _update_lru(self, key: str):
        """Update LRU status for memory cache"""
        hashed_key = self._hash_key(key)

        # Remove key from current position (if exists)
        if hashed_key in self.memory_lru:
            self.memory_lru.remove(hashed_key)

        # Add to front of LRU list
        self.memory_lru.insert(0, hashed_key)

        # Evict if over size
        if len(self.memory_lru) > self.memory_size:
            evict_key = self.memory_lru.pop()
            if evict_key in self.memory:
                del self.memory[evict_key]
                logger.debug(f"Evicted key from memory cache: {evict_key}")

    async def get(self, key: str, model: str = "default") -> Optional[Any]:
        """Get value from cache with multi-tier lookup

        Args:
            key: Cache key
            model: Model name for metrics

        Returns:
            Cached value or None if not found
        """
        hashed_key = self._hash_key(key)

        # Try memory cache first (fastest)
        if hashed_key in self.memory:
            value, expiry = self.memory[hashed_key]

            # Check if expired
            if expiry is None or time.time() < expiry:
                self._update_lru(key)  # Update LRU status
                self.stats["memory_hits"] += 1
                logger.debug(f"Memory cache hit: {hashed_key}")
                return value
            else:
                # Remove expired item
                del self.memory[hashed_key]
                if hashed_key in self.memory_lru:
                    self.memory_lru.remove(hashed_key)

        # Try Redis cache if available
        if self.redis:
            try:
                redis_value = await self.redis.get(hashed_key)
                if redis_value:
                    value_dict = json.loads(redis_value)

                    # Check TTL
                    if (
                        "expiry" not in value_dict
                        or value_dict["expiry"] is None
                        or time.time() < value_dict["expiry"]
                    ):
                        # Cache in memory for future
                        self.memory[hashed_key] = (
                            value_dict["value"],
                            value_dict.get("expiry"),
                        )
                        self._update_lru(key)

                        self.stats["redis_hits"] += 1
                        logger.debug(f"Redis cache hit: {hashed_key}")
                        return value_dict["value"]
            except Exception as e:
                logger.error(f"Redis cache error: {e}")

        # Try disk cache
        if self.disk_path:
            disk_cache_path = os.path.join(self.disk_path, f"{hashed_key}.json")
            if os.path.exists(disk_cache_path):
                try:
                    with open(disk_cache_path, "r") as f:
                        value_dict = json.load(f)

                    # Check TTL
                    if (
                        "expiry" not in value_dict
                        or value_dict["expiry"] is None
                        or time.time() < value_dict["expiry"]
                    ):
                        # Cache in memory for future
                        self.memory[hashed_key] = (
                            value_dict["value"],
                            value_dict.get("expiry"),
                        )
                        self._update_lru(key)

                        self.stats["disk_hits"] += 1
                        logger.debug(f"Disk cache hit: {hashed_key}")
                        return value_dict["value"]
                    else:
                        # Remove expired file
                        os.remove(disk_cache_path)
                except Exception as e:
                    logger.error(f"Disk cache error: {e}")

        # Cache miss
        self.stats["misses"] += 1
        logger.debug(f"Cache miss: {hashed_key}")
        return None

    async def set(
        self, key: str, value: Any, model: str = "default", ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with multi-tier storage

        Args:
            key: Cache key
            value: Value to store
            model: Model name for metrics
            ttl: Time-to-live in seconds (None for no expiry)

        Returns:
            True if set successfully, False otherwise
        """
        hashed_key = self._hash_key(key)
        expiry = (
            time.time() + (ttl or self.ttl)
            if ttl is not None or self.ttl is not None
            else None
        )

        # Set in memory cache
        self.memory[hashed_key] = (value, expiry)
        self._update_lru(key)

        # Prepare value dict for Redis and disk
        value_dict = {
            "value": value,
            "expiry": expiry,
            "model": model,
            "timestamp": time.time(),
        }

        # Set in Redis if available
        if self.redis:
            try:
                await self.redis.set(
                    hashed_key,
                    json.dumps(value_dict),
                    ex=(
                        ttl or self.ttl
                        if ttl is not None or self.ttl is not None
                        else None
                    ),
                )
            except Exception as e:
                logger.error(f"Redis cache set error: {e}")

        # Set in disk cache
        if self.disk_path:
            disk_cache_path = os.path.join(self.disk_path, f"{hashed_key}.json")
            try:
                with open(disk_cache_path, "w") as f:
                    json.dump(value_dict, f)
            except Exception as e:
                logger.error(f"Disk cache set error: {e}")

        self.stats["sets"] += 1
        logger.debug(f"Cache set: {hashed_key}")
        return True

    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels

        Args:
            key: Cache key

        Returns:
            True if deleted from at least one level, False otherwise
        """
        hashed_key = self._hash_key(key)
        deleted = False

        # Delete from memory
        if hashed_key in self.memory:
            del self.memory[hashed_key]
            if hashed_key in self.memory_lru:
                self.memory_lru.remove(hashed_key)
            deleted = True

        # Delete from Redis
        if self.redis:
            try:
                await self.redis.delete(hashed_key)
                deleted = True
            except Exception as e:
                logger.error(f"Redis cache delete error: {e}")

        # Delete from disk
        if self.disk_path:
            disk_cache_path = os.path.join(self.disk_path, f"{hashed_key}.json")
            if os.path.exists(disk_cache_path):
                try:
                    os.remove(disk_cache_path)
                    deleted = True
                except Exception as e:
                    logger.error(f"Disk cache delete error: {e}")

        return deleted

    async def clear(self, level: Optional[str] = None) -> bool:
        """Clear cache at specified level or all levels

        Args:
            level: Cache level to clear ("memory", "redis", "disk", or None for all)

        Returns:
            True if cleared successfully, False otherwise
        """
        if level is None or level == "memory":
            self.memory = {}
            self.memory_lru = []
            logger.info("Memory cache cleared")

        if (level is None or level == "redis") and self.redis:
            try:
                await self.redis.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.error(f"Redis cache clear error: {e}")
                return False

        if (level is None or level == "disk") and self.disk_path:
            try:
                for item in os.listdir(self.disk_path):
                    if item.endswith(".json"):
                        os.remove(os.path.join(self.disk_path, item))
                logger.info("Disk cache cleared")
            except Exception as e:
                logger.error(f"Disk cache clear error: {e}")
                return False

        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Dict with cache statistics
        """
        total_hits = sum(v for k, v in self.stats.items() if k.endswith("_hits"))
        total_ops = total_hits + self.stats["misses"]

        if total_ops > 0:
            hit_rate = total_hits / total_ops
        else:
            hit_rate = 0

        stats = {
            **self.stats,
            "total_hits": total_hits,
            "total_ops": total_ops,
            "hit_rate": hit_rate,
            "memory_items": len(self.memory),
            "memory_size_limit": self.memory_size,
        }

        # Add Redis stats if available
        if self.redis:
            try:
                redis_info = await self.redis.info()
                stats["redis_memory_used"] = redis_info.get("used_memory_human")
                stats["redis_keys"] = await self.redis.dbsize()
            except Exception as e:
                logger.error(f"Redis stats error: {e}")

        # Add disk stats
        if self.disk_path:
            try:
                disk_files = [
                    f for f in os.listdir(self.disk_path) if f.endswith(".json")
                ]
                stats["disk_items"] = len(disk_files)
                stats["disk_size"] = sum(
                    os.path.getsize(os.path.join(self.disk_path, f)) for f in disk_files
                )
            except Exception as e:
                logger.error(f"Disk stats error: {e}")

        return stats
