"""
调仓计算器 — 目标组合 vs 当前持仓 → 交易清单。
"""

from decimal import Decimal


class Rebalancer:
    """调仓计算器。"""

    def __init__(
        self,
        min_trade_amount: Decimal = Decimal("0.005"),  # 最小调仓阈值 (0.5%)
    ):
        self.min_trade_amount = min_trade_amount

    # ── 调仓计算 ────────────────────────────────────────

    def compute_trades(
        self,
        target_portfolio: list[dict],         # [{code, weight}]
        current_positions: dict[str, dict],   # {code: {weight, quantity, ...}}
        total_capital: Decimal,
    ) -> dict:
        """计算调仓清单（目标 vs 当前）。

        Returns:
            {buys: [{code, target_weight, current_weight, delta_weight, amount, quantity}],
             sells: [...], holds: [...]}
        """
        buys = []
        sells = []
        holds = []

        target_map = {p["code"]: p["weight"] for p in target_portfolio}
        all_codes = set(target_map.keys()) | set(current_positions.keys())

        for code in all_codes:
            target_w = target_map.get(code, Decimal("0"))
            current_info = current_positions.get(code, {})
            current_w = Decimal(str(current_info.get("weight", 0)))
            delta = target_w - current_w

            if abs(delta) < self.min_trade_amount:
                holds.append({"code": code, "target_weight": target_w, "current_weight": current_w, "delta": delta})
            elif delta > 0:
                buys.append({
                    "code": code,
                    "target_weight": target_w.quantize(Decimal("0.0001")),
                    "current_weight": current_w.quantize(Decimal("0.0001")),
                    "delta_weight": delta.quantize(Decimal("0.0001")),
                    "amount": (delta * total_capital).quantize(Decimal("0.01")),
                })
            else:
                sells.append({
                    "code": code,
                    "target_weight": target_w.quantize(Decimal("0.0001")),
                    "current_weight": current_w.quantize(Decimal("0.0001")),
                    "delta_weight": delta.quantize(Decimal("0.0001")),
                    "amount": (abs(delta) * total_capital).quantize(Decimal("0.01")),
                })

        # 按调仓金额降序
        buys.sort(key=lambda x: x["amount"], reverse=True)
        sells.sort(key=lambda x: x["amount"], reverse=True)

        total_turnover = Decimal("0")
        total_turnover += sum((abs(b["delta_weight"]) for b in buys), Decimal("0"))
        total_turnover += sum((abs(s["delta_weight"]) for s in sells), Decimal("0"))

        return {
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "total_turnover": total_turnover.quantize(Decimal("0.0001")),
            "buy_count": len(buys),
            "sell_count": len(sells),
            "hold_count": len(holds),
        }
