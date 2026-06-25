"""绩效指标计算: 夏普/回撤/胜率/年化收益/Calmar/IC/IR。"""
import math
from decimal import Decimal
from typing import Optional


class PerformanceMetrics:
    @staticmethod
    def compute_all(equity_curve: list[Decimal], daily_returns: list[Decimal], risk_free: Decimal = Decimal("0.03")) -> dict:
        if len(equity_curve) < 2:
            return {"total_return": 0.0, "annualized_return": None, "annualized_vol": None, "sharpe": None, "max_drawdown": 0.0, "calmar": None, "win_rate": 0.0, "n_days": 0}
        total_return = (equity_curve[-1] / equity_curve[0] - Decimal("1")) if equity_curve[0] > 0 else Decimal("0")
        ann_return = PerformanceMetrics._annualized_return(daily_returns)
        ann_vol = PerformanceMetrics._annualized_vol(daily_returns)
        sharpe = PerformanceMetrics.sharpe_ratio(daily_returns, risk_free)
        max_dd = PerformanceMetrics.max_drawdown(equity_curve)
        calmar = ann_return / max_dd if max_dd and max_dd > 0 else None
        win_rate = PerformanceMetrics.win_rate(daily_returns)
        return {"total_return": float(total_return.quantize(Decimal("0.0001"))), "annualized_return": float(ann_return.quantize(Decimal("0.0001"))) if ann_return else None, "annualized_vol": float(ann_vol.quantize(Decimal("0.0001"))) if ann_vol else None, "sharpe": float(sharpe.quantize(Decimal("0.0001"))) if sharpe else None, "max_drawdown": float(max_dd.quantize(Decimal("0.0001"))), "calmar": float(calmar.quantize(Decimal("0.0001"))) if calmar else None, "win_rate": float(win_rate.quantize(Decimal("0.0001"))), "n_days": len(daily_returns)}

    @staticmethod
    def sharpe_ratio(daily_returns: list[Decimal], risk_free: Decimal = Decimal("0.03")) -> Optional[Decimal]:
        if len(daily_returns) < 20: return None
        ann_ret = PerformanceMetrics._annualized_return(daily_returns)
        ann_vol = PerformanceMetrics._annualized_vol(daily_returns)
        if ann_vol is None or ann_vol == 0: return None
        return (ann_ret - risk_free) / ann_vol

    @staticmethod
    def max_drawdown(equity_curve: list[Decimal]) -> Decimal:
        peak, max_dd = equity_curve[0], Decimal("0")
        for v in equity_curve:
            if v > peak: peak = v
            dd = (peak - v) / peak if peak > 0 else Decimal("0")
            if dd > max_dd: max_dd = dd
        return max_dd

    @staticmethod
    def win_rate(daily_returns: list[Decimal]) -> Decimal:
        if not daily_returns: return Decimal("0")
        return Decimal(str(sum(1 for r in daily_returns if r > 0))) / Decimal(str(len(daily_returns)))

    @staticmethod
    def _annualized_return(daily_returns: list[Decimal]) -> Optional[Decimal]:
        if len(daily_returns) < 20: return None
        n = Decimal(len(daily_returns)); mean = sum(daily_returns) / n
        return mean * Decimal("252")

    @staticmethod
    def _annualized_vol(daily_returns: list[Decimal]) -> Optional[Decimal]:
        if len(daily_returns) < 20: return None
        n = Decimal(len(daily_returns)); mean = sum(daily_returns) / n
        var = sum((r - mean) ** 2 for r in daily_returns) / n
        return var.sqrt() * Decimal("252").sqrt()
