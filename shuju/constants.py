"""
shuju 模块可配置常量（已弃用）。

P0-2: 所有配置已迁移到 shuju.config.ShujuConfig。
本文件保留向后兼容的模块级变量，内部转发到 ShujuConfig 单例。

推荐用法:
    from shuju.config import get_config
    cfg = get_config()
    max_retries = cfg.max_retries
"""

import warnings

from shuju.config import get_config as _get_config

_cfg = _get_config()

# ── 请求控制 ──────────────────────────────────────────
MAX_RETRIES = _cfg.max_retries
REQUEST_INTERVAL = _cfg.request_interval
BATCH_REQUEST_INTERVAL = 0.05  # 保留硬编码，ShujuConfig 中未定义

# ── 预处理 ────────────────────────────────────────────
WINSORIZE_METHOD = _cfg.winsorize_method
WINSORIZE_N_SIGMA = float(_cfg.n_sigma)
FILL_METHOD = _cfg.fill_method
MIN_SAMPLE_SIZE = _cfg.min_sample_size

# ── 对齐 ──────────────────────────────────────────────
FINANCIAL_FILL_LIMIT_DAYS = _cfg.fill_limit_days

# ── 缓存 TTL (秒) ─────────────────────────────────────
TTL_DAILY_BAR = _cfg.ttl_daily_bar
TTL_FINANCIAL = _cfg.ttl_financial
TTL_NEWS = _cfg.ttl_news
TTL_SENTIMENT = _cfg.ttl_sentiment
TTL_PREPROCESSED = _cfg.ttl_preprocessed
TTL_INDUSTRY = _cfg.ttl_industry

# ── 舆情 ──────────────────────────────────────────────
SENTIMENT_NEGATION_WINDOW = 5
SENTIMENT_SAMPLE_CODES = [
    "000001", "000002", "000858", "002415", "002594",
    "300750", "600036", "600519", "601012", "601318",
]
