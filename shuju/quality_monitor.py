"""
数据质量监控器 (v4.1).

完整实现:
    - 连接 DataAligner 输出 → 自动检查缺失率
    - 每日数据拉取后自动运行质量检查
    - 异常检测: 单日数据量 < 历史均值的 50% → 告警
    - 生成每日数据质量报告 (JSON)

Usage:
    from shuju.quality_monitor import DataQualityMonitor
    monitor = DataQualityMonitor()
    report = monitor.run_quality_checks(aligned_data_dict, expected_columns, date_str)
    monitor.save_report(report)
"""

import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from shuju.utils import safe_divide, safe_mean

_logger = logging.getLogger(__name__)

# 默认报告输出目录
_DEFAULT_REPORT_DIR = Path(__file__).parent.parent / "data" / "quality_reports"

# 异常检测阈值
_ANOMALY_COUNT_RATIO: Decimal = Decimal("0.5")  # < 50% 历史均值


class DataQualityMonitor:
    """数据质量自动化监控.

    整合 completeness/freshness/distribution/duplicates 检查,
    支持 DataAligner 输出的自动检查和每日报告生成。
    """

    def __init__(self, report_dir: str | Path | None = None) -> None:
        """初始化质量监控器.

        Args:
            report_dir: 质量报告输出目录, 默认 data/quality_reports/
        """
        self._report_dir = Path(report_dir) if report_dir else _DEFAULT_REPORT_DIR
        self._report_dir.mkdir(parents=True, exist_ok=True)
        # 历史日记录计数: {date_str: record_count}
        self._history: dict[str, int] = {}
        self._load_history()

    # ══════════════════════════════════════════════════════════
    # 历史统计
    # ══════════════════════════════════════════════════════════

    def _load_history(self) -> None:
        """从最近的报告中恢复历史统计."""
        if not self._report_dir.exists():
            return
        try:
            for report_file in sorted(self._report_dir.glob("quality_*.json")):
                try:
                    with open(report_file, encoding="utf-8") as f:
                        data = json.load(f)
                    date_str = data.get("date", "")
                    total = data.get("total_records", 0)
                    if date_str and total > 0:
                        self._history[date_str] = total
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception as exc:
            _logger.debug("加载历史报告失败: %s", exc)

    def _get_historical_mean(self) -> Decimal | None:
        """计算历史日均记录数."""
        if not self._history:
            return None
        values = [Decimal(v) for v in self._history.values()]
        return safe_mean(values)

    # ══════════════════════════════════════════════════════════
    # 完整性检查
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def check_completeness(
        column_values: list[str | None],
        column_name: str = "unknown",
    ) -> tuple[bool, str]:
        """检查缺失值比例.

        Args:
            column_values: 列值列表
            column_name: 列名 (用于报告)

        Returns:
            (通过?, 描述信息)
        """
        n = len(column_values)
        if n == 0:
            return False, f"{column_name}: empty column"
        missing = sum(1 for v in column_values if v is None)
        ratio = Decimal(missing) / Decimal(n)
        ok = ratio < Decimal("0.05")
        return ok, f"{column_name}: {missing}/{n} missing ({float(ratio):.1%})"

    @staticmethod
    def check_completeness_dict(
        data: dict[str, list[Any]],
        expected_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """检查 DataAligner 输出字典中各列的缺失率.

        Args:
            data: {column_name: [values], ...} 对齐后的数据字典
            expected_columns: 期望存在的列 (可选)

        Returns:
            {
                "overall_pass": bool,
                "total_records": int,
                "column_results": {col: {"pass": bool, "missing": int, "total": int, "ratio": float}},
            }
        """
        if not data:
            return {"overall_pass": False, "total_records": 0, "column_results": {}}

        col_results: dict[str, Any] = {}
        all_pass = True
        max_len = 0

        columns = expected_columns or list(data.keys())
        for col in columns:
            values = data.get(col, [])
            if not values:
                col_results[col] = {"pass": False, "missing": 0, "total": 0, "ratio": 0.0}
                all_pass = False
                continue
            n = len(values)
            max_len = max(max_len, n)
            missing = sum(1 for v in values if v is None)
            ratio = float(Decimal(missing) / Decimal(n)) if n > 0 else 1.0
            ok = ratio < 0.05
            col_results[col] = {"pass": ok, "missing": missing, "total": n, "ratio": ratio}
            if not ok:
                all_pass = False

        return {
            "overall_pass": all_pass,
            "total_records": max_len,
            "column_results": col_results,
        }

    # ══════════════════════════════════════════════════════════
    # 时效性检查
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def check_freshness(
        last_date: str,
        max_age_days: int = 3,
    ) -> tuple[bool, str]:
        """检查数据时效性.

        Args:
            last_date: 最新数据日期 (YYYY-MM-DD 或 YYYYMMDD)
            max_age_days: 最大允许延迟天数

        Returns:
            (通过?, 描述信息)
        """
        try:
            # 标准化日期
            if len(last_date) == 8 and last_date.isdigit():
                last = datetime.strptime(last_date, "%Y%m%d").date()
            else:
                last = datetime.strptime(last_date[:10], "%Y-%m-%d").date()
            today = datetime.now().date()
            age = (today - last).days
            ok = age <= max_age_days
            return ok, f"Last date: {last_date} ({age}d old, max: {max_age_days}d)"
        except (ValueError, IndexError):
            return False, f"Invalid date: {last_date}"

    # ══════════════════════════════════════════════════════════
    # 分布检查
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def check_distribution(
        values: list[Decimal],
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> tuple[bool, str]:
        """检查数值分布是否在合理区间.

        Args:
            values: 数值列表
            lower_bound: 下界 (μ - 3σ 不应低于此值)
            upper_bound: 上界 (μ + 3σ 不应高于此值)

        Returns:
            (通过?, 描述信息)
        """
        n = len(values)
        if n == 0:
            return False, "No values to check"

        mu = safe_mean(values)
        var = sum((v - mu) ** 2 for v in values) / Decimal(max(1, n - 1))
        sigma = var.sqrt()

        in_range = True
        if lower_bound is not None and float(mu - Decimal("3") * sigma) < lower_bound:
            in_range = False
        if upper_bound is not None and float(mu + Decimal("3") * sigma) > upper_bound:
            in_range = False

        return in_range, f"μ={float(mu):.4f} σ={float(sigma):.4f} n={n}"

    # ══════════════════════════════════════════════════════════
    # 重复检查
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def check_duplicates(
        keys: list[str],
    ) -> tuple[bool, str]:
        """检查重复键.

        Args:
            keys: 键列表

        Returns:
            (通过?, 描述信息)
        """
        n = len(keys)
        unique = len(set(keys))
        dup = n - unique
        ok = dup == 0
        return ok, f"{dup} duplicates in {n} keys"

    # ══════════════════════════════════════════════════════════
    # 异常检测
    # ══════════════════════════════════════════════════════════

    def check_anomaly(
        self,
        record_count: int,
        date_str: str,
    ) -> tuple[bool, str]:
        """检测今日数据量与历史均值相比是否出现异常.

        异常条件: 单日数据量 < 历史均值的 50%

        Args:
            record_count: 今日记录数
            date_str: 日期字符串

        Returns:
            (正常?, 描述信息)
        """
        hist_mean = self._get_historical_mean()
        if hist_mean is None or hist_mean == 0:
            return True, f"First run: {record_count} records (no history)"

        ratio = safe_divide(Decimal(record_count), hist_mean)
        ok = ratio >= _ANOMALY_COUNT_RATIO
        status = "正常" if ok else "异常"
        return ok, (
            f"{status}: {record_count} records vs historical avg {float(hist_mean):.0f} "
            f"(ratio: {float(ratio):.2f}, threshold: 0.50)"
        )

    # ══════════════════════════════════════════════════════════
    # 综合质量检查
    # ══════════════════════════════════════════════════════════

    def run_quality_checks(
        self,
        aligned_data: dict[str, list[Any]],
        expected_columns: list[str] | None = None,
        date_str: str | None = None,
        last_date: str | None = None,
        anomaly_check: bool = True,
    ) -> dict[str, Any]:
        """运行全量质量检查.

        Args:
            aligned_data: DataAligner 输出格式的数据字典
                {col_name: [values], ...}
                支持嵌套: {"code_value": {col: [values]}}
            expected_columns: 期望列
            date_str: 数据日期 (YYYY-MM-DD)
            last_date: 最后数据更新日期
            anomaly_check: 是否运行异常检测

        Returns:
            完整质量报告 dict
        """
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")

        # Flatten: 如果数据是按 code 分组的, 展开
        flat_data = self._flatten_data(aligned_data)

        # 1. 完整性
        completeness = self.check_completeness_dict(flat_data, expected_columns)

        # 2. 重复检查 (trade_date + code 组合键)
        dup_ok, dup_msg = True, "skipped"
        if "code" in flat_data and "trade_date" in flat_data:
            keys = [f"{c}_{d}" for c, d in zip(flat_data["code"], flat_data["trade_date"], strict=False)]
            dup_ok, dup_msg = self.check_duplicates(keys)

        # 3. 时效性
        fresh_ok, fresh_msg = True, "no last_date provided"
        if last_date:
            fresh_ok, fresh_msg = self.check_freshness(last_date)

        # 4. 分布检查 (数值列)
        dist_results: dict[str, Any] = {}
        numeric_cols = [
            col for col in flat_data
            if col in ("open", "high", "low", "close", "volume", "amount",
                       "pe", "pb", "roe", "roa", "return", "factor_value")
        ]
        for col in numeric_cols:
            values = flat_data.get(col, [])
            decimals = _to_decimal_list(values)
            if decimals:
                dist_ok, dist_msg = self.check_distribution(decimals)
                if not dist_ok:
                    dist_results[col] = {"pass": dist_ok, "message": dist_msg}

        # 5. 异常检测
        anomaly_ok, anomaly_msg = True, "skipped"
        if anomaly_check:
            anomaly_ok, anomaly_msg = self.check_anomaly(
                completeness["total_records"], date_str
            )

        # 汇总
        all_pass = (
            completeness["overall_pass"]
            and dup_ok
            and fresh_ok
            and anomaly_ok
        )

        return {
            "date": date_str,
            "timestamp": datetime.now().isoformat(),
            "overall_pass": all_pass,
            "total_records": completeness["total_records"],
            "completeness": completeness,
            "duplicates": {"pass": dup_ok, "message": dup_msg},
            "freshness": {"pass": fresh_ok, "message": fresh_msg},
            "distribution_warnings": dist_results,
            "anomaly": {"pass": anomaly_ok, "message": anomaly_msg},
            "checks_summary": {
                "completeness": completeness["overall_pass"],
                "duplicates": dup_ok,
                "freshness": fresh_ok,
                "anomaly": anomaly_ok,
                "distribution_ok": len(dist_results) == 0,
            },
        }

    # ══════════════════════════════════════════════════════════
    # 报告持久化
    # ══════════════════════════════════════════════════════════

    def save_report(self, report: dict[str, Any]) -> str:
        """保存质量报告到 JSON 文件.

        Args:
            report: run_quality_checks() 返回的报告 dict

        Returns:
            报告文件路径
        """
        date_str = report.get("date", datetime.now().strftime("%Y-%m-%d"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_{date_str}_{timestamp}.json"
        filepath = self._report_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # 更新历史记录
        self._history[date_str] = report.get("total_records", 0)

        _logger.info("质量报告已保存: %s", filepath)
        return str(filepath)

    def get_latest_report(self) -> dict[str, Any] | None:
        """获取最近的质量报告."""
        reports = sorted(self._report_dir.glob("quality_*.json"))
        if not reports:
            return None
        try:
            with open(reports[-1], encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    # ══════════════════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _flatten_data(
        data: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """展开按 code 分组的嵌套数据.

        检测数据是否形如 {"code": [...], "close": [...]} (已 flat)
        或 {"000001": {"close": [...]}, "000002": {...}} (嵌套).

        Args:
            data: 可能是 flat 或嵌套的数据字典

        Returns:
            扁平化的列字典
        """
        if not data:
            return {}

        # 检测格式: 如果所有 key 都是 6 位数字 → 嵌套格式
        nested_keys = [k for k in data if isinstance(k, str) and len(k) == 6 and k.isdigit()]
        if nested_keys and len(nested_keys) > len(data) * 0.5:
            # 嵌套格式: 按列聚合
            flat: dict[str, list[Any]] = {}
            for _code, code_data in data.items():
                if isinstance(code_data, dict):
                    for col, values in code_data.items():
                        if col not in flat:
                            flat[col] = []
                        if isinstance(values, list):
                            flat[col].extend(values)
                        else:
                            flat[col].append(values)
            return flat

        # 已是 flat 格式
        return {k: v for k, v in data.items() if isinstance(v, list)}


def _to_decimal_list(values: list[Any]) -> list[Decimal]:
    """将任意列表转换为 Decimal 列表, 跳过无效值."""
    result: list[Decimal] = []
    for v in values:
        if v is None:
            continue
        try:
            if isinstance(v, Decimal):
                result.append(v)
            else:
                result.append(Decimal(str(v)))
        except Exception:
            pass
    return result
