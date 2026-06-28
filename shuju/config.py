"""
shuju 统一配置。

所有可调参数集中管理，支持环境变量覆盖。

环境变量:
    SHUJU_MAX_RETRIES          — 最大重试次数 (默认 3)
    SHUJU_REQUEST_INTERVAL     — 请求间隔秒数 (默认 0.5)
    SHUJU_N_SIGMA              — 去极值 sigma 倍数 (默认 3)
    SHUJU_FILL_LIMIT_DAYS      — 前值填充最大天数 (默认 90)
    SHUJU_TTL_DAILY_BAR        — 日线缓存 TTL 秒 (默认 14400)
    SHUJU_TTL_FINANCIAL        — 财务缓存 TTL 秒 (默认 86400)
    SHUJU_TTL_NEWS             — 新闻缓存 TTL 秒 (默认 1800)
    SHUJU_TTL_SENTIMENT        — 舆情缓存 TTL 秒 (默认 1800)
    SHUJU_TTL_PREPROCESSED     — 预处理缓存 TTL 秒 (默认 7200)
    SHUJU_TTL_INDUSTRY         — 行业缓存 TTL 秒 (默认 86400)
    SHUJU_CACHE_KEY_PREFIX     — 缓存键前缀 (默认 lingshu:data:)
"""

import os
from dataclasses import dataclass, field
from decimal import Decimal

# ── 财务因子字段名 (TushareFetcher 与 DataAligner 共享) ────

FINANCIAL_FIELD_NAMES: tuple[str, ...] = (
    "pe", "pb", "ps", "roe", "roa", "gross_margin",
    "net_margin", "revenue", "net_profit",
    "operating_cashflow", "free_cashflow_yield",
)

# ── 默认样本股票 (SentimentFetcher 市场情绪) ──────────────

DEFAULT_SAMPLE_CODES: list[str] = [
    "000001", "000002", "000858", "002415", "002594",
    "300750", "600036", "600519", "601012", "601318",
]


@dataclass
class ShujuConfig:
    """shuju 数据层统一配置。

    所有字段均有合理默认值，可通过构造函数或环境变量覆盖。
    """

    # ── 重试与频率控制 ─────────────────────────────────
    max_retries: int = 3
    request_interval: float = 0.5

    # ── 预处理 ─────────────────────────────────────────
    winsorize_method: str = "mad"            # "mad" | "sigma"
    n_sigma: Decimal = field(default_factory=lambda: Decimal("3"))
    mad_scale: Decimal = field(default_factory=lambda: Decimal("0.6745"))
    fill_method: str = "median"              # "median" | "zero"
    min_sample_size: int = 5

    # ── 对齐器 ─────────────────────────────────────────
    fill_limit_days: int = 90

    # ── 缓存 TTL (秒) ──────────────────────────────────
    ttl_daily_bar: int = 4 * 3600           # 4 小时
    ttl_financial: int = 24 * 3600          # 24 小时
    ttl_news: int = 30 * 60                 # 30 分钟
    ttl_sentiment: int = 30 * 60            # 30 分钟
    ttl_preprocessed: int = 2 * 3600        # 2 小时
    ttl_industry: int = 24 * 3600           # 24 小时

    # ── 缓存基础 ───────────────────────────────────────
    cache_key_prefix: str = "lingshu:data:"

    # ── 情感分析阈值 ───────────────────────────────────
    sentiment_strong_positive: Decimal = field(default_factory=lambda: Decimal("0.3"))
    sentiment_mild_positive: Decimal = field(default_factory=lambda: Decimal("0.1"))
    sentiment_mild_negative: Decimal = field(default_factory=lambda: Decimal("-0.1"))
    sentiment_strong_negative: Decimal = field(default_factory=lambda: Decimal("-0.3"))
    market_optimistic: float = 0.2
    market_mild_optimistic: float = 0.05
    market_mild_pessimistic: float = -0.05
    market_pessimistic: float = -0.2

    @classmethod
    def from_env(cls) -> "ShujuConfig":
        """从环境变量创建配置，未设置则使用默认值。"""
        kwargs: dict = {}

        def _int(k, d):
            return int(os.getenv(k, d))
        def _float(k, d):
            return float(os.getenv(k, d))
        def _str(k, d):
            return os.getenv(k, d)

        kwargs["max_retries"] = _int("SHUJU_MAX_RETRIES", 3)
        kwargs["request_interval"] = _float("SHUJU_REQUEST_INTERVAL", 0.5)
        kwargs["winsorize_method"] = _str("SHUJU_WINSORIZE_METHOD", "mad")
        if v := os.getenv("SHUJU_N_SIGMA"):
            kwargs["n_sigma"] = Decimal(v)
        kwargs["fill_method"] = _str("SHUJU_FILL_METHOD", "median")
        kwargs["fill_limit_days"] = _int("SHUJU_FILL_LIMIT_DAYS", 90)
        kwargs["ttl_daily_bar"] = _int("SHUJU_TTL_DAILY_BAR", 4 * 3600)
        kwargs["ttl_financial"] = _int("SHUJU_TTL_FINANCIAL", 24 * 3600)
        kwargs["ttl_news"] = _int("SHUJU_TTL_NEWS", 30 * 60)
        kwargs["ttl_sentiment"] = _int("SHUJU_TTL_SENTIMENT", 30 * 60)
        kwargs["ttl_preprocessed"] = _int("SHUJU_TTL_PREPROCESSED", 2 * 3600)
        kwargs["ttl_industry"] = _int("SHUJU_TTL_INDUSTRY", 24 * 3600)
        kwargs["cache_key_prefix"] = _str("SHUJU_CACHE_KEY_PREFIX", "lingshu:data:")

        return cls(**kwargs)


# ── 模块级单例 ────────────────────────────────────────────

_config: ShujuConfig | None = None


def get_config() -> ShujuConfig:
    """获取当前配置单例（首次调用时从环境变量加载）。"""
    global _config
    if _config is None:
        _config = ShujuConfig.from_env()
    return _config


def set_config(config: ShujuConfig) -> None:
    """替换全局配置（供测试使用）。"""
    global _config
    _config = config


def reset_config() -> None:
    """重置为环境变量默认（供测试拆解）。"""
    global _config
    _config = None
