"""
shuju 共享工具函数。

消除 akshare_fetcher 和 tushare_fetcher 之间的重复代码。
所有金融计算必须使用 safe_divide / safe_mean / safe_pct_change 替代裸运算符。
"""

from decimal import Decimal, DecimalException
from typing import Any, Union

import pandas as pd

Number = Union[int, float, Decimal]


# ── 安全数学运算 (v4.0: 从 shujuku/safe_divide.py 复活) ──────


def safe_divide(
    numerator: Number,
    denominator: Number,
    default: Number = Decimal("0"),
) -> Decimal:
    """安全除法：分母为零/None/无效值时返回默认值。"""
    if denominator is None or numerator is None:
        return Decimal(str(default))
    try:
        num = Decimal(str(numerator))
        den = Decimal(str(denominator))
        return Decimal(str(default)) if den == 0 else num / den
    except DecimalException:
        return Decimal(str(default))


def safe_mean(values: list[Number], default: Decimal = Decimal("0")) -> Decimal:
    """安全均值：空列表或全无效值时返回默认值。"""
    if not values:
        return default
    decimals = []
    for v in values:
        try:
            decimals.append(Decimal(str(v)))
        except DecimalException:
            pass
    if not decimals:
        return default
    return sum(decimals) / Decimal(len(decimals))


def safe_pct_change(old: Number, new: Number, default: Decimal = Decimal("0")) -> Decimal:
    """安全涨跌幅：(new - old) / old。old 为零时返回默认值。"""
    try:
        o = Decimal(str(old))
        n = Decimal(str(new))
        return default if o == 0 else (n - o) / o
    except DecimalException:
        return default


def safe_decimal(value: Any) -> Decimal | None:
    """安全转换数值为 Decimal（处理 NaN/None/非数值字符串）。

    AKShareFetcher._to_decimal 和 TushareFetcher._safe_decimal 的统一实现。
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value in ("-", "—", "", "nan", "NaN", "None"):
            return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


# ── 共享重试装饰器 ──────────────────────────────────────

def make_retry(source_name: str, max_retries: int = 3, logger=None):
    """创建带 Prometheus 指标的重试装饰器。

    Args:
        source_name: 数据源标识 (akshare/tushare)
        max_retries: 最大重试次数
        logger: 日志记录器

    Returns:
        装饰器函数
    """
    import functools
    import time as _time

    from shuju.metrics import fetcher_latency, fetcher_requests_total, fetcher_retries_total

    def retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = func.__name__.replace("_fetch_", "").replace("_raw", "")
            start = _time.perf_counter()
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    fetcher_requests_total.labels(source=source_name, operation=op_name, status="success").inc()
                    fetcher_latency.labels(source=source_name, operation=op_name).observe(_time.perf_counter() - start)
                    return result
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        fetcher_retries_total.labels(source=source_name, operation=op_name).inc()
                        wait = 2 ** attempt
                        if logger:
                            logger.warning("%s attempt %d/%d failed: %s, retrying in %ds",
                                          func.__name__, attempt, max_retries, exc, wait)
                        _time.sleep(wait)
            fetcher_requests_total.labels(source=source_name, operation=op_name, status="failure").inc()
            if logger:
                logger.error("%s all %d attempts failed: %s", func.__name__, max_retries, last_exc)
            return None
        return wrapper
    return retry
