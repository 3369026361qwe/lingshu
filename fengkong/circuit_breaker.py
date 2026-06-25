"""三态熔断器 — L1 防护层。CLOSED ↔ OPEN ↔ HALF_OPEN 自动切换。"""
import time as _time
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal


class BreakerState(str, Enum):
    CLOSED = "CLOSED"         # 正常
    OPEN = "OPEN"             # 熔断
    HALF_OPEN = "HALF_OPEN"   # 半开（试探恢复）


class CircuitBreaker:
    """三态熔断器。"""

    def __init__(self, max_consecutive_losses: int = 3, max_daily_loss: Decimal = Decimal("0.02"), max_drawdown: Decimal = Decimal("0.08"), cooldown_seconds: int = 300):
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.cooldown_seconds = cooldown_seconds
        self._state = BreakerState.CLOSED
        self._consecutive_losses = 0
        self._daily_pnl = Decimal("0")
        self._peak_equity = Decimal("0")
        self._current_equity = Decimal("0")
        self._opened_at: float = 0.0
        self._history: list[dict] = []

    @property
    def state(self) -> BreakerState:
        self._check_cooldown()
        return self._state

    def update(self, daily_pnl: Decimal, current_equity: Decimal) -> BreakerState:
        """输入当日盈亏和当前权益，自动判断状态切换。"""
        self._current_equity = current_equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        drawdown = (self._peak_equity - current_equity) / self._peak_equity if self._peak_equity > 0 else Decimal("0")

        if daily_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        should_open = (self._consecutive_losses >= self.max_consecutive_losses or abs(daily_pnl) / max(current_equity, Decimal("1")) >= self.max_daily_loss or drawdown >= self.max_drawdown)

        if self._state == BreakerState.CLOSED and should_open:
            self._transition(BreakerState.OPEN, f"触发熔断: 连续亏损={self._consecutive_losses}, 日亏损={daily_pnl}, 回撤={drawdown}")
        elif self._state == BreakerState.HALF_OPEN and daily_pnl >= 0:
            self._transition(BreakerState.CLOSED, "试探成功，恢复交易")
        elif self._state == BreakerState.HALF_OPEN and daily_pnl < 0:
            self._transition(BreakerState.OPEN, "试探失败，重新熔断")

        return self._state

    def reset(self) -> None:
        self._state = BreakerState.CLOSED
        self._consecutive_losses = 0
        self._daily_pnl = Decimal("0")

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def is_blocked(self) -> bool:
        return self.state == BreakerState.OPEN

    def _transition(self, to_state: BreakerState, reason: str) -> None:
        self._history.append({"from": self._state.value, "to": to_state.value, "reason": reason, "time": datetime.now(timezone.utc).isoformat()})
        self._state = to_state
        if to_state == BreakerState.OPEN:
            self._opened_at = _time.time()

    def _check_cooldown(self) -> None:
        if self._state == BreakerState.OPEN and _time.time() - self._opened_at >= self.cooldown_seconds:
            self._state = BreakerState.HALF_OPEN
            self._history.append({"from": "OPEN", "to": "HALF_OPEN", "reason": "冷却时间到，进入试探期", "time": datetime.now(timezone.utc).isoformat()})
