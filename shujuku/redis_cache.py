"""
Redis 缓存管理器。

提供可选的 Redis 缓存层，Redis 不可用时自动降级到内存 dict 缓存。
所有操作静默降级，不抛异常。

Usage:
    cache = CacheManager()
    cache.set("key", "value", ttl=300)
    value = cache.get("key")  # 返回 str | None
"""

import json
import logging
import threading
import time
from typing import Any, Optional

from shujuku.config import REDIS_URL
from shujuku.metrics import cache_hits, cache_misses, redis_available as redis_available_gauge, cache_backend_active

_logger = logging.getLogger(__name__)

# Redis 重连冷却时间（秒）
_RECONNECT_COOLDOWN = 30


class _MemoryStore:
    """内存缓存后端（Redis 不可用时的降级方案）。"""

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at > 0 and time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: str, ttl: int = 0) -> None:
        with self._lock:
            expires_at = time.time() + ttl if ttl > 0 else 0
            self._store[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class CacheManager:
    """缓存管理器。线程安全。

    优先级: Redis > 内存 dict
    Redis 连接失败 → 静默切换到内存模式
    Redis 恢复 → 定时重连后自动切回
    """

    def __init__(self, key_prefix: str = "lingshu:") -> None:
        self._prefix = key_prefix
        self._redis = None
        self._fallback = _MemoryStore()
        self._redis_available = False
        self._state_lock = threading.Lock()          # C3: 保护状态转换
        self._last_reconnect_attempt: float = 0.0    # H5: 重连冷却

        self._init_redis()
        self._update_gauges()

    # ── 内部方法 ────────────────────────────────────────

    def _update_gauges(self) -> None:
        """同步 Prometheus 指标状态。"""
        redis_available_gauge.set(1 if self._redis_available else 0)
        cache_backend_active.set(1 if self._redis_available else 0)

    def _init_redis(self) -> None:
        """尝试连接 Redis。失败则保持降级模式。"""
        if not REDIS_URL:
            return
        try:
            import redis
            self._redis = redis.Redis.from_url(
                REDIS_URL,
                socket_connect_timeout=3,
                socket_timeout=3,
                decode_responses=True,
            )
            self._redis.ping()
            self._redis_available = True
        except Exception:
            self._redis = None
            self._redis_available = False

    def _mark_redis_unavailable(self) -> None:
        """标记 Redis 不可用并更新指标（需持有 _state_lock）。"""
        self._redis_available = False
        self._update_gauges()

    def _try_reconnect(self) -> bool:
        """H5: 尝试重连 Redis。成功返回 True。受冷却时间限制。"""
        now = time.time()
        if now - self._last_reconnect_attempt < _RECONNECT_COOLDOWN:
            return False
        self._last_reconnect_attempt = now
        if not REDIS_URL:
            return False
        try:
            import redis
            self._redis = redis.Redis.from_url(
                REDIS_URL,
                socket_connect_timeout=3,
                socket_timeout=3,
                decode_responses=True,
            )
            self._redis.ping()
            self._redis_available = True
            self._update_gauges()
            _logger.info("Redis reconnected successfully")
            return True
        except Exception:
            self._redis = None
            return False

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    # ── 公共接口 ────────────────────────────────────────

    def get(self, key: str) -> Optional[str]:
        """获取缓存值。"""
        full_key = self._full_key(key)
        # Redis 路径
        with self._state_lock:
            available = self._redis_available
        if available and self._redis:
            try:
                result = self._redis.get(full_key)
                backend = "redis"
                if result is not None:
                    cache_hits.labels(backend=backend).inc()
                else:
                    cache_misses.labels(backend=backend).inc()
                return result
            except Exception:
                with self._state_lock:
                    self._mark_redis_unavailable()
        # 降级到内存
        result = self._fallback.get(full_key)
        if result is not None:
            cache_hits.labels(backend="memory").inc()
        else:
            cache_misses.labels(backend="memory").inc()
        # 尝试后台重连
        if not available:
            with self._state_lock:
                self._try_reconnect()
        return result

    def set(self, key: str, value: str, ttl: int = 0) -> None:
        """设置缓存值。ttl <= 0 表示永不过期。"""
        full_key = self._full_key(key)
        with self._state_lock:
            available = self._redis_available
        if available and self._redis:
            try:
                if ttl > 0:
                    self._redis.setex(full_key, ttl, value)
                else:
                    self._redis.set(full_key, value)
                return
            except Exception:
                with self._state_lock:
                    self._mark_redis_unavailable()
        self._fallback.set(full_key, value, ttl)
        if not available:
            with self._state_lock:
                self._try_reconnect()

    def get_json(self, key: str) -> Optional[Any]:
        """获取并解析 JSON 缓存值。"""
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_json(self, key: str, value: Any, ttl: int = 0) -> bool:
        """序列化为 JSON 后缓存。返回 True 表示成功。"""
        try:
            raw = json.dumps(value, ensure_ascii=False, default=str)
            self.set(key, raw, ttl)
            return True
        except (TypeError, ValueError) as exc:
            _logger.warning("set_json failed for key=%s: %s", key, exc)
            return False

    def delete(self, key: str) -> None:
        """删除缓存键。"""
        full_key = self._full_key(key)
        with self._state_lock:
            available = self._redis_available
        if available and self._redis:
            try:
                self._redis.delete(full_key)
                return
            except Exception:
                with self._state_lock:
                    self._mark_redis_unavailable()
        self._fallback.delete(full_key)

    def clear(self) -> None:
        """清空所有带前缀的缓存。"""
        # H3: 使用 SCAN 代替 KEYS，避免阻塞 Redis 服务器
        with self._state_lock:
            available = self._redis_available
        if available and self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(
                        cursor, match=f"{self._prefix}*", count=100
                    )
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception:
                with self._state_lock:
                    self._mark_redis_unavailable()
        # LOW-5: 始终清理内存降级缓存，防止 Redis 宕机后读到过期数据
        self._fallback.clear()

    @property
    def is_redis_available(self) -> bool:
        """Redis 是否可用。"""
        return self._redis_available
