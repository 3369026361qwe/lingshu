"""
数据层缓存管理器。

封装 shujuku.CacheManager，提供数据场景专用的缓存策略：
    - 日线行情: TTL 4小时（交易日每天更新一次）
    - 财务数据: TTL 24小时（季频更新）
    - 新闻舆情: TTL 30分钟（实时变化）
    - 预处理结果: TTL 2小时

Usage:
    cache = DataCacheManager()
    cache.cache_daily_bar("000001", "2026-06-01", bar_dict)
    bar = cache.get_daily_bar("000001", "2026-06-01")
"""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from shujuku.redis_cache import CacheManager
from shuju.constants import (
    TTL_DAILY_BAR, TTL_FINANCIAL, TTL_NEWS, TTL_SENTIMENT,
    TTL_PREPROCESSED, TTL_INDUSTRY,
)
from shuju.metrics import data_cache_hits, data_cache_misses


# ── Decimal 安全序列化 ──────────────────────────────────
# CRITICAL FIX: json.dumps 将 Decimal→str 后，json.loads 不会自动还原为 Decimal。
# 使用 {__decimal__: "value"} 标记 + 自定义 decoder 保证往返类型安全。

_DECIMAL_MARKER = "__decimal__"


class _DecimalEncoder(json.JSONEncoder):
    """JSON 编码器: Decimal → {__decimal__: str}, date → ISO string。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return {_DECIMAL_MARKER: str(obj)}
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def _decimal_decoder(dct: dict) -> dict:
    """JSON 解码器: {__decimal__: str} → Decimal。"""
    for key, value in dct.items():
        if isinstance(value, dict) and _DECIMAL_MARKER in value:
            dct[key] = Decimal(value[_DECIMAL_MARKER])
    return dct


class DataCacheManager:
    """数据层缓存管理器，封装通用 CacheManager 提供领域语义。"""

    def __init__(self) -> None:
        self._cache = CacheManager(key_prefix="lingshu:data:")

    # ── 日线行情 ────────────────────────────────────────

    # ── 内部序列化 ──────────────────────────────────────

    def _safe_serialize(self, data: Any) -> str:
        """CRITICAL: Decimal 安全的 JSON 序列化。"""
        return json.dumps(data, ensure_ascii=False, cls=_DecimalEncoder)

    def _safe_deserialize(self, raw: str) -> Any:
        """CRITICAL: Decimal 安全的 JSON 反序列化（含异常保护）。"""
        try:
            return json.loads(raw, object_hook=_decimal_decoder)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            import logging
            logging.getLogger(__name__).warning("Cache deserialization failed: %s", exc)
            return None

    # ── 日线行情 ────────────────────────────────────────

    def cache_daily_bar(self, code: str, trade_date: str, bar: dict) -> None:
        """缓存单条日线数据。"""
        key = f"bar:{code}:{trade_date}"
        self._cache.set(key, self._safe_serialize(bar), ttl=TTL_DAILY_BAR)

    def _cache_get(self, key: str, data_type: str) -> Optional[str]:
        """内部: 获取缓存原始值并记录命中/未命中指标。"""
        raw = self._cache.get(key)
        if raw:
            data_cache_hits.labels(data_type=data_type).inc()
        else:
            data_cache_misses.labels(data_type=data_type).inc()
        return raw

    def get_daily_bar(self, code: str, trade_date: str) -> Optional[dict]:
        """获取缓存的日线数据。"""
        key = f"bar:{code}:{trade_date}"
        raw = self._cache_get(key, "daily_bar")
        return self._safe_deserialize(raw) if raw else None

    def cache_daily_bars_batch(self, code: str, bars: list[dict]) -> None:
        """批量缓存某股票的全部日线。"""
        key = f"bars:{code}"
        self._cache.set(key, self._safe_serialize(bars), ttl=TTL_DAILY_BAR)

    def get_daily_bars_batch(self, code: str) -> Optional[list[dict]]:
        """获取某股票的全部缓存日线。"""
        key = f"bars:{code}"
        raw = self._cache_get(key, "daily_bar")
        return self._safe_deserialize(raw) if raw else None

    # ── 财务数据 ────────────────────────────────────────

    def cache_financial(self, code: str, report_date: str, data: dict) -> None:
        """缓存单条财务数据。"""
        key = f"fin:{code}:{report_date}"
        self._cache.set(key, self._safe_serialize(data), ttl=TTL_FINANCIAL)

    def get_financial(self, code: str, report_date: str) -> Optional[dict]:
        """获取缓存的财务数据。"""
        key = f"fin:{code}:{report_date}"
        raw = self._cache_get(key, "financial")
        return self._safe_deserialize(raw) if raw else None

    # ── 新闻舆情 ────────────────────────────────────────

    def cache_news(self, news_id: str, data: dict) -> None:
        """缓存单条新闻。"""
        key = f"news:{news_id}"
        self._cache.set(key, self._safe_serialize(data), ttl=TTL_NEWS)

    def get_news(self, news_id: str) -> Optional[dict]:
        """获取缓存的新闻。"""
        key = f"news:{news_id}"
        raw = self._cache_get(key, "news")
        return self._safe_deserialize(raw) if raw else None

    def cache_sentiment(self, code: str, data: dict) -> None:
        """缓存单只股票舆情。"""
        key = f"sentiment:{code}"
        self._cache.set(key, self._safe_serialize(data), ttl=TTL_SENTIMENT)

    def get_sentiment(self, code: str) -> Optional[dict]:
        """获取缓存的舆情数据。"""
        key = f"sentiment:{code}"
        raw = self._cache_get(key, "sentiment")
        return self._safe_deserialize(raw) if raw else None

    # ── 预处理结果 ──────────────────────────────────────

    def cache_preprocessed(self, dataset_id: str, data: Any) -> None:
        """缓存预处理后的数据集。"""
        key = f"pp:{dataset_id}"
        self._cache.set(key, self._safe_serialize(data), ttl=TTL_PREPROCESSED)

    def get_preprocessed(self, dataset_id: str) -> Optional[Any]:
        """获取缓存的预处理结果。"""
        key = f"pp:{dataset_id}"
        raw = self._cache_get(key, "preprocessed")
        return self._safe_deserialize(raw) if raw else None

    # ── 行业分类 ────────────────────────────────────────

    def cache_industry(self, code: str, data: dict) -> None:
        """缓存行业分类。"""
        key = f"industry:{code}"
        self._cache.set(key, self._safe_serialize(data), ttl=TTL_INDUSTRY)

    def get_industry(self, code: str) -> Optional[dict]:
        """获取缓存的行业分类。"""
        key = f"industry:{code}"
        raw = self._cache_get(key, "industry")
        return self._safe_deserialize(raw) if raw else None

    # ── 通用 ────────────────────────────────────────────

    def clear_all(self) -> None:
        """清空所有数据层缓存。"""
        self._cache.clear()

    @property
    def is_redis_available(self) -> bool:
        return self._cache.is_redis_available
