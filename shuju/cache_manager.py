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
from typing import Any

from shuju.config import get_config

_CACHE = get_config()
from shuju.metrics import data_cache_hit_rate, data_cache_hits, data_cache_misses
from shujuku.redis_cache import CacheManager

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
    """数据层缓存管理器，封装通用 CacheManager 提供领域语义。

    缓存命中率通过 Prometheus Gauge 实时暴露:
        lingshu_data_cache_hit_rate{data_type="daily_bar"} 0.85
    """

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
        self._cache.set(key, self._safe_serialize(bar), ttl=_CACHE.ttl_daily_bar)

    def _cache_get(self, key: str, data_type: str) -> str | None:
        """内部: 获取缓存原始值并记录命中/未命中指标。"""
        raw = self._cache.get(key)
        if raw:
            data_cache_hits.labels(data_type=data_type).inc()
        else:
            data_cache_misses.labels(data_type=data_type).inc()
        return raw

    def get_daily_bar(self, code: str, trade_date: str) -> dict | None:
        """获取缓存的日线数据。"""
        key = f"bar:{code}:{trade_date}"
        raw = self._cache_get(key, "daily_bar")
        return self._safe_deserialize(raw) if raw else None

    def cache_daily_bars_batch(self, code: str, bars: list[dict]) -> None:
        """批量缓存某股票的全部日线。"""
        key = f"bars:{code}"
        self._cache.set(key, self._safe_serialize(bars), ttl=_CACHE.ttl_daily_bar)

    def get_daily_bars_batch(self, code: str) -> list[dict] | None:
        """获取某股票的全部缓存日线。"""
        key = f"bars:{code}"
        raw = self._cache_get(key, "daily_bar")
        return self._safe_deserialize(raw) if raw else None

    # ── 财务数据 ────────────────────────────────────────

    def cache_financial(self, code: str, report_date: str, data: dict) -> None:
        """缓存单条财务数据。"""
        key = f"fin:{code}:{report_date}"
        self._cache.set(key, self._safe_serialize(data), ttl=_CACHE.ttl_financial)

    def get_financial(self, code: str, report_date: str) -> dict | None:
        """获取缓存的财务数据。"""
        key = f"fin:{code}:{report_date}"
        raw = self._cache_get(key, "financial")
        return self._safe_deserialize(raw) if raw else None

    # ── 新闻舆情 ────────────────────────────────────────

    def cache_news(self, news_id: str, data: dict) -> None:
        """缓存单条新闻。"""
        key = f"news:{news_id}"
        self._cache.set(key, self._safe_serialize(data), ttl=_CACHE.ttl_news)

    def get_news(self, news_id: str) -> dict | None:
        """获取缓存的新闻。"""
        key = f"news:{news_id}"
        raw = self._cache_get(key, "news")
        return self._safe_deserialize(raw) if raw else None

    def cache_sentiment(self, code: str, data: dict) -> None:
        """缓存单只股票舆情。"""
        key = f"sentiment:{code}"
        self._cache.set(key, self._safe_serialize(data), ttl=_CACHE.ttl_sentiment)

    def get_sentiment(self, code: str) -> dict | None:
        """获取缓存的舆情数据。"""
        key = f"sentiment:{code}"
        raw = self._cache_get(key, "sentiment")
        return self._safe_deserialize(raw) if raw else None

    # ── 预处理结果 ──────────────────────────────────────

    def cache_preprocessed(self, dataset_id: str, data: Any) -> None:
        """缓存预处理后的数据集。"""
        key = f"pp:{dataset_id}"
        self._cache.set(key, self._safe_serialize(data), ttl=_CACHE.ttl_preprocessed)

    def get_preprocessed(self, dataset_id: str) -> Any | None:
        """获取缓存的预处理结果。"""
        key = f"pp:{dataset_id}"
        raw = self._cache_get(key, "preprocessed")
        return self._safe_deserialize(raw) if raw else None

    # ── 行业分类 ────────────────────────────────────────

    def cache_industry(self, code: str, data: dict) -> None:
        """缓存行业分类。"""
        key = f"industry:{code}"
        self._cache.set(key, self._safe_serialize(data), ttl=_CACHE.ttl_industry)

    def get_industry(self, code: str) -> dict | None:
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

    def hit_rate(self, data_type: str) -> float:
        """获取某类缓存的当前命中率.

        Args:
            data_type: 缓存类型 (daily_bar/financial/news/sentiment/industry/preprocessed)

        Returns:
            命中率 (0.0 ~ 1.0), 无数据时返回 -1.0
        """
        try:
            from prometheus_client import REGISTRY

            hits = REGISTRY.get_sample_value(
                "lingshu_data_cache_hits_total", {"data_type": data_type}
            ) or 0.0
            misses = REGISTRY.get_sample_value(
                "lingshu_data_cache_misses_total", {"data_type": data_type}
            ) or 0.0
            total = hits + misses
            if total == 0:
                return -1.0
            return float(hits / total)
        except Exception:
            return -1.0


# ══════════════════════════════════════════════════════════════
# 模块级辅助: 缓存命中率计算 (v4.1)
# ══════════════════════════════════════════════════════════════

def _update_hit_rate(data_type: str) -> None:
    """从 hits/misses Counter 实时计算并更新命中率 Gauge."""
    # 使用 prometheus_client REGISTRY 的 get_sample_value() 获取计数值
    try:
        from prometheus_client import REGISTRY

        hits_val = REGISTRY.get_sample_value(
            "lingshu_data_cache_hits_total", {"data_type": data_type}
        ) or 0.0
        misses_val = REGISTRY.get_sample_value(
            "lingshu_data_cache_misses_total", {"data_type": data_type}
        ) or 0.0
        total = hits_val + misses_val
        if total > 0:
            data_cache_hit_rate.labels(data_type=data_type).set(hits_val / total)
    except Exception:
        pass  # 指标收集失败时静默跳过
