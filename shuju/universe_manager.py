"""
股票宇宙管理 — Point-in-Time + 幸存者偏差治理 (v4.1).

完整实现:
    - 从 AKShare 获取退市股票列表, 维护 data/delisted_stocks.csv
    - Point-in-Time 股票宇宙 (含已退市股票)
    - ST/*ST 实时过滤 (从 name 字段检测)
    - 停牌检测 (当日成交量=0 且非涨跌停)

Usage:
    from shuju.universe_manager import UniverseManager
    manager = UniverseManager()
    manager.fetch_delisted_from_akshare()  # 首次初始化
    universe = manager.survivorship_free_universe("2024-01-15")
"""

import csv
import logging
from datetime import date as _date
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

# 默认退市股票数据库路径
_DEFAULT_DELISTED_PATH = Path(__file__).parent.parent / "data" / "delisted_stocks.csv"


class UniverseManager:
    """Point-in-Time 股票宇宙管理.

    维护退市股票历史和上市日期, 确保回测时使用正确的股票列表,
    消除幸存者偏差。

    Attributes:
        _delist_db: {code: (list_date, delist_date)} — 退市股票记录
        _active_names: {code: name} — 当前活跃股票名称 (用于 ST 检测)
        _active_list_dates: {code: list_date} — 活跃股票上市日期
    """

    def __init__(self, delisted_csv_path: str | Path | None = None) -> None:
        """初始化 UniverseManager.

        Args:
            delisted_csv_path: 退市股票 CSV 路径, 默认 data/delisted_stocks.csv
        """
        self._csv_path = Path(delisted_csv_path) if delisted_csv_path else _DEFAULT_DELISTED_PATH
        # {code: (list_date, delist_date)} — list_date/delist_date 均为 YYYY-MM-DD
        self._delist_db: dict[str, tuple[str, str | None]] = {}
        # {code: name} — 当前活跃股票
        self._active_names: dict[str, str] = {}
        # {code: list_date} — 活跃股票上市日期
        self._active_list_dates: dict[str, str] = {}
        self._load_delisted()

    # ══════════════════════════════════════════════════════════
    # CSV 持久化
    # ══════════════════════════════════════════════════════════

    def _load_delisted(self) -> None:
        """从 CSV 文件加载退市股票数据库."""
        if not self._csv_path.exists():
            _logger.info("退市股票文件不存在: %s, 从空数据库开始", self._csv_path)
            return
        try:
            with open(self._csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    code = (row.get("code") or "").strip()
                    list_date = (row.get("list_date") or "").strip()
                    delist = (row.get("delist_date") or "").strip()
                    delist_date: str | None = delist if delist else None
                    if code and list_date and len(code) == 6:
                        self._delist_db[code] = (list_date, delist_date)
                        count += 1
            _logger.info("已加载 %d 条退市股票记录", count)
        except Exception as exc:
            _logger.warning("退市股票加载失败: %s", exc)

    def _save_delisted(self) -> None:
        """保存退市股票数据库到 CSV."""
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["code", "list_date", "delist_date"])
            for code in sorted(self._delist_db):
                list_date, delist_date = self._delist_db[code]
                writer.writerow([code, list_date, delist_date or ""])
        _logger.info("已保存 %d 条退市股票记录到 %s", len(self._delist_db), self._csv_path)

    # ══════════════════════════════════════════════════════════
    # AKShare 数据同步
    # ══════════════════════════════════════════════════════════

    def fetch_delisted_from_akshare(self) -> int:
        """从 AKShare 获取退市股票列表并更新数据库.

        Returns:
            新增的退市股票数量. 如果 AKShare 不可用则返回 0.
        """
        try:
            import akshare as ak
        except ImportError:
            _logger.warning("AKShare 不可用, 跳过退市股票获取")
            return 0

        new_count = 0
        try:
            df = ak.stock_zh_a_stop_info()
            if df is None or df.empty:
                _logger.warning("AKShare 返回空退市列表")
                return 0

            for _, row in df.iterrows():
                try:
                    code = str(row.get("code", "")).zfill(6)
                    if not code or len(code) != 6:
                        continue
                    list_date = _normalize_date(str(row.get("list_date", "")))
                    stop_date_raw = str(row.get("stop_date", ""))
                    delist_date = _normalize_date(stop_date_raw) if stop_date_raw else None

                    if not list_date:
                        continue
                    if code not in self._delist_db:
                        self._delist_db[code] = (list_date, delist_date)
                        new_count += 1
                except Exception:
                    continue

            if new_count > 0:
                self._save_delisted()
            _logger.info("从 AKShare 获取了 %d 只新退市股票", new_count)
        except Exception as exc:
            _logger.warning("AKShare 退市股票获取失败: %s", exc)

        return new_count

    def fetch_active_stocks_from_akshare(self) -> dict[str, str]:
        """从 AKShare 获取当前活跃股票列表 (含名称).

        Returns:
            {code: name} 字典
        """
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            if df is None or df.empty:
                return {}
            result: dict[str, str] = {}
            for _, row in df.iterrows():
                try:
                    code = str(row.get("code", "")).zfill(6)
                    name = str(row.get("name", ""))
                    if code and name and len(code) == 6:
                        result[code] = name
                except Exception:
                    continue
            self._active_names = result
            _logger.info("从 AKShare 获取了 %d 只活跃股票", len(result))
            return result
        except ImportError:
            _logger.warning("AKShare 不可用")
            return {}
        except Exception as exc:
            _logger.warning("活跃股票获取失败: %s", exc)
            return {}

    # ══════════════════════════════════════════════════════════
    # Point-in-Time 宇宙
    # ══════════════════════════════════════════════════════════

    def survivorship_free_universe(
        self,
        query_date: str,
        include_delisted: bool = True,
    ) -> list[str]:
        """返回给定日期的完整股票列表 (Point-in-Time, 含已退市).

        逻辑:
        1. 从活跃股票中筛选出 query_date 时已上市的
        2. 从退市数据库中提取 query_date 时仍在交易且之后退市的
        3. 并集去重排序

        Args:
            query_date: 查询日期 (YYYY-MM-DD)
            include_delisted: 是否包含已退市股票 (默认 True)

        Returns:
            股票代码列表, 按代码排序
        """
        qd = _date.fromisoformat(query_date)
        codes: set[str] = set()

        # 1. 当前活跃股票中, query_date 时已上市的
        for code in self._active_names:
            list_date_str = self._active_list_dates.get(code)
            if list_date_str:
                try:
                    if _date.fromisoformat(list_date_str) > qd:
                        continue  # 尚未上市
                except ValueError:
                    pass
            codes.add(code)

        # 2. 退市股票 (include_delisted=True 时)
        if include_delisted:
            for code, (list_d, delist_d) in self._delist_db.items():
                try:
                    list_date = _date.fromisoformat(list_d)
                    if list_date > qd:
                        continue  # 尚未上市
                    if delist_d is not None:
                        if _date.fromisoformat(delist_d) < qd:
                            continue  # 已经退市
                    codes.add(code)
                except (ValueError, TypeError):
                    continue

        return sorted(codes)

    # ══════════════════════════════════════════════════════════
    # ST / *ST 过滤
    # ══════════════════════════════════════════════════════════

    def filter_st_star(
        self,
        universe: list[str],
        stock_names: dict[str, str] | None = None,
    ) -> list[str]:
        """过滤 ST / *ST 股票.

        优先使用 stock_names 参数的名称映射, 其次使用内部缓存的 _active_names.
        两者都没有时回退到从代码检测 (保守策略).

        Args:
            universe: 股票代码列表
            stock_names: {code: name} 映射 (可选)

        Returns:
            过滤后的股票代码列表
        """
        names = stock_names or self._active_names
        if names:
            return [code for code in universe if not _is_st_name(names.get(code, ""))]
        # Fallback: 从代码检测
        return [code for code in universe if "ST" not in code.upper()]

    # ══════════════════════════════════════════════════════════
    # 停牌检测
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def filter_suspended(
        universe: list[str],
        daily_data: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        """过滤当日停牌股票.

        判断逻辑:
        - 成交量 = 0 且不是一字涨跌停板 → 停牌
        - 一字板 (涨/跌停): open=high=low=close, 涨跌幅约 ±9.5%~10.5%

        Args:
            universe: 股票代码列表
            daily_data: {code: {volume, open, high, low, close, pre_close}} 字典

        Returns:
            过滤后的股票代码列表
        """
        if not daily_data:
            return list(universe)

        result: list[str] = []
        for code in universe:
            bar = daily_data.get(code)
            if bar is None:
                # 无数据 → 保守保留
                result.append(code)
                continue

            volume = _to_float(bar.get("volume", 0))
            if volume > 0:
                result.append(code)
                continue

            # 成交量 = 0: 检查是否一字板
            if _is_limit_board(bar):
                result.append(code)  # 一字板保留
            # 否则 → 停牌, 过滤掉

        return result

    # ══════════════════════════════════════════════════════════
    # 程序化注册 (测试/离线场景)
    # ══════════════════════════════════════════════════════════

    def register_delisted(self, code: str, list_date: str, delist_date: str | None) -> None:
        """注册退市股票信息.

        Args:
            code: 6位股票代码
            list_date: 上市日期 (YYYY-MM-DD)
            delist_date: 退市日期 (YYYY-MM-DD), None 表示尚未退市
        """
        self._delist_db[code] = (list_date, delist_date)

    def register_active_stocks(self, stocks: dict[str, str]) -> None:
        """注册活跃股票列表 (供离线/测试使用).

        Args:
            stocks: {code: name} 字典
        """
        self._active_names = dict(stocks)

    def register_active_list_dates(self, dates: dict[str, str]) -> None:
        """注册活跃股票上市日期 (供离线/测试使用).

        Args:
            dates: {code: list_date (YYYY-MM-DD)} 字典
        """
        self._active_list_dates = dict(dates)

    def save(self) -> None:
        """持久化退市数据库到 CSV."""
        self._save_delisted()

    # ══════════════════════════════════════════════════════════
    # 统计属性
    # ══════════════════════════════════════════════════════════

    @property
    def delisted_count(self) -> int:
        """退市股票数量."""
        return len(self._delist_db)

    @property
    def active_count(self) -> int:
        """活跃股票数量."""
        return len(self._active_names)

    @property
    def delisted_codes(self) -> list[str]:
        """退市股票代码列表."""
        return sorted(self._delist_db)


# ══════════════════════════════════════════════════════════════
# 模块级辅助函数
# ══════════════════════════════════════════════════════════════


def _is_st_name(name: str) -> bool:
    """判断股票名称是否表示 ST/*ST 股票.

    Args:
        name: 股票名称 (如 "ST瑞德", "*ST华英", "平安银行")

    Returns:
        True 如果名称以 ST 或 *ST 开头
    """
    if not name:
        return False
    upper = name.strip().upper()
    return upper.startswith("ST") or upper.startswith("*ST")


def _to_float(value: Any) -> float:
    """安全转换为 float, 失败时返回 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_limit_board(bar: dict[str, Any]) -> bool:
    """判断是否为一字涨跌停板.

    一字板特征: open=high=low=close, 且涨跌幅约 ±9.5%~10.5%.

    Args:
        bar: {open, high, low, close, pre_close} 字典

    Returns:
        True 如果是一字板 (涨停或跌停)
    """
    o = _to_float(bar.get("open", 0))
    h = _to_float(bar.get("high", 0))
    lo = _to_float(bar.get("low", 0))
    c = _to_float(bar.get("close", 0))
    pre = _to_float(bar.get("pre_close", 0))

    # 价格不能全为零
    if any(v == 0.0 for v in (o, h, lo, c, pre)):
        return False

    # 一字: O=H=L=C
    if not (_fuzzy_eq(o, h) and _fuzzy_eq(o, lo) and _fuzzy_eq(o, c)):
        return False

    # 涨跌幅判断 (支持 ±5%, ±10%, ±20% 不同板块)
    pct_change = abs((c - pre) / pre)
    return 0.045 <= pct_change <= 0.22


def _fuzzy_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    """浮点模糊相等."""
    return abs(a - b) < tol


def _normalize_date(raw: str) -> str:
    """标准化日期格式为 YYYY-MM-DD.

    支持: YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD
    """
    raw = raw.strip()
    if not raw:
        return ""
    # YYYYMMDD
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    # 已经是 YYYY-MM-DD
    return raw.replace("/", "-")
