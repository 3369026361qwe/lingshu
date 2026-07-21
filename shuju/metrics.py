"""
shuju 数据层 Prometheus 指标。

复用 shujuku.metrics 的 REGISTRY，指标名统一使用 lingshu_ 前缀。
"""

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# ── 数据采集指标 ─────────────────────────────────────

fetcher_requests_total = Counter(
    "lingshu_fetcher_requests_total",
    "Total data fetcher requests",
    ["source", "operation", "status"],  # source=akshare/tushare/news, status=success/failure
    registry=REGISTRY,
)

fetcher_latency = Histogram(
    "lingshu_fetcher_latency_seconds",
    "Data fetcher request latency",
    ["source", "operation"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

fetcher_retries_total = Counter(
    "lingshu_fetcher_retries_total",
    "Total fetcher retry attempts",
    ["source", "operation"],
    registry=REGISTRY,
)

# ── 预处理指标 ───────────────────────────────────────

preprocessor_duration = Histogram(
    "lingshu_preprocessor_duration_seconds",
    "Preprocessing pipeline duration",
    ["stage"],  # winsorize / fill_missing / standardize / neutralize / pipeline
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=REGISTRY,
)

preprocessor_records_total = Counter(
    "lingshu_preprocessor_records_total",
    "Total records processed",
    ["stage"],
    registry=REGISTRY,
)

# ── 对齐器指标 ───────────────────────────────────────

aligner_duration = Histogram(
    "lingshu_aligner_duration_seconds",
    "Data alignment duration",
    ["method"],  # financial_to_daily / sentiment_to_daily / align_to_daily
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY,
)

aligner_records_total = Counter(
    "lingshu_aligner_records_total",
    "Total aligned records",
    ["method"],
    registry=REGISTRY,
)

# ── 数据缓存指标 ─────────────────────────────────────

data_cache_hits = Counter(
    "lingshu_data_cache_hits_total",
    "Data cache hits",
    ["data_type"],  # daily_bar / financial / news / sentiment / industry / preprocessed
    registry=REGISTRY,
)

data_cache_misses = Counter(
    "lingshu_data_cache_misses_total",
    "Data cache misses",
    ["data_type"],
    registry=REGISTRY,
)

# ── 缓存命中率监控 (v4.1) ─────────────────────────────

data_cache_hit_rate = Gauge(
    "lingshu_data_cache_hit_rate",
    "Cache hit rate (0.0 ~ 1.0) per data type",
    ["data_type"],
    registry=REGISTRY,
)

data_cache_total_ops = Counter(
    "lingshu_data_cache_ops_total",
    "Total cache operations (hits + misses) for hit rate calculation",
    ["data_type"],
    registry=REGISTRY,
)

# ── 情感分析指标 ─────────────────────────────────────

sentiment_analyzed_total = Counter(
    "lingshu_sentiment_analyzed_total",
    "Total sentiment analyses performed",
    ["type"],  # stock / market
    registry=REGISTRY,
)
