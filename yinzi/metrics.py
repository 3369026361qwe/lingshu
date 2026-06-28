"""
yinzi 因子引擎 Prometheus 指标。

复用全局 REGISTRY，指标名统一使用 lingshu_ 前缀。
"""

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# ── 因子计算指标 ─────────────────────────────────────

factor_compute_total = Counter(
    "lingshu_factor_compute_total",
    "Total factor computations",
    ["factor_name", "status"],  # status=success/failure/skipped
    registry=REGISTRY,
)

factor_compute_duration = Histogram(
    "lingshu_factor_compute_duration_seconds",
    "Factor computation duration",
    ["factor_name"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    registry=REGISTRY,
)

factor_batch_duration = Histogram(
    "lingshu_factor_batch_duration_seconds",
    "Full batch factor computation duration",
    ["category"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)

# ── 因子有效性指标 ───────────────────────────────────

factor_ic = Gauge(
    "lingshu_factor_ic",
    "Latest Rank IC value",
    ["factor_name"],
    registry=REGISTRY,
)

factor_ir = Gauge(
    "lingshu_factor_ir",
    "Latest Information Ratio",
    ["factor_name"],
    registry=REGISTRY,
)

factor_coverage = Gauge(
    "lingshu_factor_coverage_ratio",
    "Factor coverage ratio (valid stocks / total stocks)",
    ["factor_name"],
    registry=REGISTRY,
)

# ── 卡尔曼滤波指标 ───────────────────────────────────

kalman_update_total = Counter(
    "lingshu_kalman_update_total",
    "Total Kalman filter updates",
    registry=REGISTRY,
)

kalman_converged = Gauge(
    "lingshu_kalman_converged",
    "Whether Kalman filter has converged (1=yes, 0=no)",
    registry=REGISTRY,
)

kalman_max_variance = Gauge(
    "lingshu_kalman_max_variance",
    "Maximum weight variance (diagonal of P matrix)",
    registry=REGISTRY,
)
