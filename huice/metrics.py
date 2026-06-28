"""huice 回测 Prometheus 指标。"""
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

backtest_runs_total = Counter("lingshu_backtest_runs_total", "Total backtest runs", registry=REGISTRY)
backtest_duration = Histogram("lingshu_backtest_duration_seconds", "Backtest duration", buckets=[1,5,10,30,60,120,300], registry=REGISTRY)
backtest_sharpe = Gauge("lingshu_backtest_sharpe", "Latest backtest Sharpe ratio", registry=REGISTRY)
backtest_max_dd = Gauge("lingshu_backtest_max_drawdown", "Latest backtest max drawdown", registry=REGISTRY)
grid_search_total = Counter("lingshu_grid_search_total", "Grid search experiments run", registry=REGISTRY)
