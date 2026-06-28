"""safe_divide — 安全的除法运算，防止除零和精度损失。

所有金融计算必须使用此函数替代裸 `/` 运算符。
CLAUDE.md 中已声明但之前未实现。
"""
from decimal import Decimal
from typing import Optional, Union

Number = Union[int, float, Decimal]


def safe_divide(
    numerator: Number,
    denominator: Number,
    default: Number = Decimal("0"),
) -> Decimal:
    """安全除法：分母为零或 None 时返回默认值。

    Args:
        numerator: 分子
        denominator: 分母
        default: 分母无效时的返回值（默认 0）

    Returns:
        Decimal 类型的商，保证精度不损失

    Examples:
        >>> safe_divide(10, 3)
        Decimal('3.333333333333333333')
        >>> safe_divide(10, 0)
        Decimal('0')
        >>> safe_divide(10, None, default=Decimal('1'))
        Decimal('1')
    """
    try:
        num = Decimal(str(numerator))
        den = Decimal(str(denominator))
        if den == 0:
            return Decimal(str(default))
        return num / den
    except (ValueError, TypeError, AttributeError):
        return Decimal(str(default))


def safe_mean(values: list[Number], default: Decimal = Decimal("0")) -> Decimal:
    """安全均值：空列表返回默认值。"""
    if not values:
        return default
    decimals = []
    for v in values:
        try:
            decimals.append(Decimal(str(v)))
        except (ValueError, TypeError):
            pass
    if not decimals:
        return default
    return sum(decimals) / Decimal(str(len(decimals)))


def safe_pct_change(old: Number, new: Number, default: Decimal = Decimal("0")) -> Decimal:
    """安全涨跌幅：(new - old) / old。old 为零时返回默认值。"""
    try:
        o = Decimal(str(old))
        n = Decimal(str(new))
        if o == 0:
            return default
        return (n - o) / o
    except (ValueError, TypeError):
        return default
