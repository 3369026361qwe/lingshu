"""
股票宇宙管理 — Point-in-Time + 幸存者偏差治理 (v4.0).

维护退市股票历史和上市日期, 确保回测时使用正确的股票列表,
消除幸存者偏差。

Usage:
    from shuju.universe_manager import UniverseManager
    universe = UniverseManager.survivorship_free_universe("2024-01-15")
"""

from datetime import date as _date


class UniverseManager:
    """Point-in-Time 股票宇宙管理.

    当前为 stub — 完整实现需要退市股票数据库。
    """

    # 退市股票记录: {code: (list_date, delist_date)}
    _delist_db: dict[str, tuple[str, str | None]] = {}

    @classmethod
    def survivorship_free_universe(
        cls, query_date: str, include_delisted: bool = True
    ) -> list[str]:
        """返回给定日期的完整股票列表 (含已退市).

        Args:
            query_date: 查询日期 (YYYY-MM-DD)
            include_delisted: 是否包含已退市股票
        Returns:
            股票代码列表
        """
        if not include_delisted or not cls._delist_db:
            return cls._active_only(query_date)
        qd = _date.fromisoformat(query_date)
        codes = cls._active_only(query_date)  # 当前活跃的
        for code, (list_d, delist_d) in cls._delist_db.items():
            list_date = _date.fromisoformat(list_d)
            if list_date <= qd:
                if delist_d is None or _date.fromisoformat(delist_d) >= qd:
                    if code not in codes:
                        codes.append(code)
        return codes

    @classmethod
    def _active_only(cls, query_date: str) -> list[str]:
        """仅返回当前仍上市交易的股票 (stub: 从数据库实时查询)."""
        # Stub: 返回空列表，由数据库层实现
        return []

    @classmethod
    def filter_suspended(cls, universe: list[str], query_date: str) -> list[str]:
        """过滤当日停牌股票.

        实时判断: 成交量 = 0 或 收盘价 = 前日收盘价 (一字板不在 filter 之列).
        Stub: 返回原列表.
        """
        return universe

    @classmethod
    def filter_st_star(cls, universe: list[str]) -> list[str]:
        """过滤 ST / *ST 股票."""
        return [code for code in universe if "ST" not in code.upper()]

    @classmethod
    def register_delisted(
        cls, code: str, list_date: str, delist_date: str | None
    ) -> None:
        """注册退市股票信息."""
        cls._delist_db[code] = (list_date, delist_date)
