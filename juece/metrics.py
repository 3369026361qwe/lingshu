"""juece 集成决策引擎 Prometheus 指标。"""
from prometheus_client import Counter, Gauge, Histogram, REGISTRY

ensemble_fusion_total = Counter("lingshu_ensemble_fusion_total", "Total fusion operations", ["mode"], registry=REGISTRY)
ensemble_fusion_duration = Histogram("lingshu_ensemble_fusion_duration_seconds", "Fusion duration", buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0], registry=REGISTRY)
selector_top_n_total = Counter("lingshu_selector_top_n_total", "Total top-N selections", registry=REGISTRY)
optimizer_optimizations_total = Counter("lingshu_optimizer_optimizations_total", "Total portfolio optimizations", registry=REGISTRY)
rebalancer_trades_total = Counter("lingshu_rebalancer_trades_total", "Total rebalance operations", registry=REGISTRY)
rebalancer_turnover = Histogram("lingshu_rebalancer_turnover_ratio", "Portfolio turnover ratio", buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0], registry=REGISTRY)
benchmark_sharpe = Gauge("lingshu_benchmark_sharpe", "Latest Sharpe ratio", registry=REGISTRY)
benchmark_max_drawdown = Gauge("lingshu_benchmark_max_drawdown", "Latest max drawdown", registry=REGISTRY)
