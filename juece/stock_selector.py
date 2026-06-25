"""
选股信号生成器。

基于综合得分生成买入/持有/卖出信号 + Top-N 选股列表。
"""

from decimal import Decimal
from typing import Optional


class StockSelector:
    """选股信号生成器。"""

    def __init__(
        self,
        top_n: int = 30,
        buy_threshold: Decimal = Decimal("0.7"),
        sell_threshold: Decimal = Decimal("0.3"),
    ):
        self.top_n = top_n
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    # ── 信号生成 ────────────────────────────────────────

    def generate_signals(
        self,
        composite_scores: dict[str, Decimal],
        current_positions: Optional[set[str]] = None,
    ) -> dict[str, dict]:
        """生成每只股票的买入/持有/卖出信号。

        Returns:
            {code: {signal: "BUY"/"HOLD"/"SELL", score: Decimal, rank: int, percentile: Decimal}}
        """
        ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked)
        if n == 0:
            return {}

        signals = {}
        for rank, (code, score) in enumerate(ranked):
            percentile = Decimal(str(rank)) / Decimal(str(n - 1)) if n > 1 else Decimal("0.5")

            if percentile <= Decimal("0.1"):  # Top 10% → BUY
                signal = "BUY"
            elif percentile >= Decimal("0.9"):  # Bottom 10% → SELL
                signal = "SELL"
            else:
                signal = "HOLD"

            signals[code] = {
                "signal": signal,
                "score": score,
                "rank": rank + 1,
                "percentile": percentile.quantize(Decimal("0.0001")),
            }

        # 对已有持仓做增强判断
        if current_positions:
            for code in current_positions:
                if code in signals:
                    s = signals[code]
                    # 如果持仓股票在 Bottom 20% → 强制 SELL
                    if s["percentile"] >= Decimal("0.8"):
                        s["signal"] = "SELL"

        return signals

    # ── Top-N 选股 ──────────────────────────────────────

    def select_top_n(
        self,
        composite_scores: dict[str, Decimal],
        exclude: Optional[set[str]] = None,
        min_score: Optional[Decimal] = None,
    ) -> list[dict]:
        """选出 Top-N 推荐股票。

        Args:
            composite_scores: 综合得分
            exclude: 排除的股票代码（已持仓/ST/涨跌停）
            min_score: 最低得分过滤

        Returns:
            [{code, score, rank, weight}]
        """
        exclude = exclude or set()
        candidates = {
            c: s for c, s in composite_scores.items()
            if c not in exclude and (min_score is None or s >= min_score)
        }
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:self.top_n]

        if not ranked:
            return []

        total_score = sum(s for _, s in ranked)
        return [
            {
                "code": code,
                "score": score.quantize(Decimal("0.0001")),
                "rank": i + 1,
                "weight": (score / total_score).quantize(Decimal("0.0001")) if total_score > 0 else Decimal("0"),
            }
            for i, (code, score) in enumerate(ranked)
        ]

    # ── 行业分散 ────────────────────────────────────────

    def diversify(
        self,
        picks: list[dict],
        industry_map: dict[str, str],
        max_per_industry: int = 3,
    ) -> list[dict]:
        """行业分散过滤：每个行业最多 max_per_industry 只。

        Args:
            picks: select_top_n 的返回值
            industry_map: {code: sw_level1}

        Returns:
            过滤后的选股列表
        """
        industry_count: dict[str, int] = {}
        result = []
        for pick in picks:
            ind = industry_map.get(pick["code"], "未知")
            if industry_count.get(ind, 0) < max_per_industry:
                result.append(pick)
                industry_count[ind] = industry_count.get(ind, 0) + 1
        return result
