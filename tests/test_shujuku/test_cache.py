"""
测试 Redis 缓存管理器（含降级模式）。
"""


from shujuku.redis_cache import CacheManager, _MemoryStore


class TestMemoryStore:
    def test_set_get(self):
        store = _MemoryStore()
        store.set("key1", "value1")
        assert store.get("key1") == "value1"

    def test_get_missing(self):
        store = _MemoryStore()
        assert store.get("nonexistent") is None

    def test_ttl_expiry(self):
        import time
        store = _MemoryStore()
        store.set("key", "value", ttl=1)
        assert store.get("key") == "value"
        time.sleep(1.1)
        assert store.get("key") is None

    def test_delete(self):
        store = _MemoryStore()
        store.set("key", "value")
        store.delete("key")
        assert store.get("key") is None

    def test_clear(self):
        store = _MemoryStore()
        store.set("a", "1")
        store.set("b", "2")
        store.clear()
        assert store.get("a") is None
        assert store.get("b") is None


class TestCacheManager:
    """默认环境下 Redis 不可用，自动降级到内存。"""

    def test_basic_set_get(self):
        cache = CacheManager(key_prefix="test:")
        cache.set("hello", "world")
        assert cache.get("hello") == "world"

    def test_json_roundtrip(self):
        cache = CacheManager(key_prefix="test:")
        data = {"name": "平安银行", "code": "000001", "score": 94.2}
        cache.set_json("stock:000001", data)
        result = cache.get_json("stock:000001")
        assert result == data

    def test_json_invalid(self):
        cache = CacheManager(key_prefix="test:")
        cache.set("invalid_json", "not-json{{{")
        result = cache.get_json("invalid_json")
        assert result is None

    def test_delete(self):
        cache = CacheManager(key_prefix="test:")
        cache.set("key", "value")
        cache.delete("key")
        assert cache.get("key") is None

    def test_redis_not_available(self):
        cache = CacheManager(key_prefix="test:")
        assert not cache.is_redis_available  # 无 REDIS_URL

    def test_prefix_isolation(self):
        cache1 = CacheManager(key_prefix="app1:")
        cache2 = CacheManager(key_prefix="app2:")
        cache1.set("key", "val1")
        cache2.set("key", "val2")
        # 使用同一个内存后端 key 包含前缀，所以各自隔离
        # 实际上两个 CacheManager 各有独立的 _MemoryStore，天然隔离
        assert cache1.get("key") == "val1"
        assert cache2.get("key") == "val2"

    def test_clear(self):
        cache = CacheManager(key_prefix="test:")
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
