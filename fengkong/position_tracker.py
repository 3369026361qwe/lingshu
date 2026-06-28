"""实时持仓追踪器。"""
from datetime import datetime, timezone
from decimal import Decimal


class PositionTracker:
    """多币种/多股票持仓追踪。"""

    def __init__(self):
        self._positions: dict[str, dict] = {}
        self._snapshots: list[dict] = []

    def update(self, code: str, quantity: int, avg_cost: Decimal, current_price: Decimal | None = None, industry: str | None = None) -> None:
        mv = Decimal(str(quantity)) * current_price if current_price else None
        self._positions[code] = {"code": code, "quantity": quantity, "avg_cost": avg_cost, "current_price": current_price, "market_value": mv, "industry": industry, "updated_at": datetime.now(timezone.utc)}

    def get(self, code: str) -> dict | None:
        return self._positions.get(code)

    def snapshot(self, total_capital: Decimal) -> list[dict]:
        """生成带权重的持仓快照。"""
        result = []
        for pos in self._positions.values():
            mv = pos.get("market_value")
            weight = mv / total_capital if mv and total_capital > 0 else Decimal("0")
            result.append({**pos, "weight": weight.quantize(Decimal("0.000001"))})
        self._snapshots.append({"time": datetime.now(timezone.utc).isoformat(), "positions": result})
        return result

    @property
    def all_positions(self) -> list[dict]:
        return list(self._positions.values())

    @property
    def position_count(self) -> int:
        return len(self._positions)

    def remove(self, code: str) -> None:
        self._positions.pop(code, None)
