"""
Prometheus 指标导出。

提供统一指标注册表和常用指标类型。
所有指标在首次导入时自动注册到 Prometheus REGISTRY。

Usage:
    from shujuku.metrics import db_ops_total, db_ops_latency

    db_ops_total.labels(operation="insert", table="factor_value").inc()
"""

from prometheus_client import Counter, Gauge, Histogram, Info, CollectorRegistry, REGISTRY

# ── 数据库操作指标 ─────────────────────────────────────

db_ops_total = Counter(
    "lingshu_db_operations_total",
    "Total database operations",
    ["operation", "table"],
    registry=REGISTRY,
)

db_ops_latency = Histogram(
    "lingshu_db_operation_latency_seconds",
    "Database operation latency in seconds",
    ["operation", "table"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

db_errors_total = Counter(
    "lingshu_db_errors_total",
    "Total database errors",
    ["operation", "error_type"],
    registry=REGISTRY,
)

# ── 缓存指标 ──────────────────────────────────────────

cache_hits = Counter(
    "lingshu_cache_hits_total",
    "Total cache hits",
    ["backend"],
    registry=REGISTRY,
)

cache_misses = Counter(
    "lingshu_cache_misses_total",
    "Total cache misses",
    ["backend"],
    registry=REGISTRY,
)

redis_available = Gauge(
    "lingshu_redis_available",
    "Whether Redis is available (1=yes, 0=no)",
    registry=REGISTRY,
)

# ── 会话指标 ──────────────────────────────────────────

session_pool_size = Gauge(
    "lingshu_db_session_pool_size",
    "Database connection pool size",
    ["state"],
    registry=REGISTRY,
)

# ── 系统信息 ──────────────────────────────────────────

_system_info = Info("lingshu_system", "LingShu system information", registry=REGISTRY)
_system_info.info({"version": "2.0.0", "module": "shujuku"})

# ── 降级状态指标 (P2-4) ────────────────────────────────

db_degraded = Gauge(
    "lingshu_db_degraded",
    "Whether the database repository is in degraded mode (1=yes, 0=no)",
    registry=REGISTRY,
)

cache_backend_active = Gauge(
    "lingshu_cache_backend_active",
    "Active cache backend (1=redis, 0=memory)",
    registry=REGISTRY,
)
