"""风控路由 — 风险状态查询 + 风险监控 WebSocket。"""

from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fengkong.risk_manager import RiskManager
from jiekou.dependencies import get_repository

router = APIRouter(prefix="/api", tags=["risk"])
ws_router = APIRouter(tags=["risk-ws"])


def _build_portfolio_from_db():
    """从数据库构建 RiskManager 所需的 portfolio 列表。

    Returns:
        (portfolio, cash, total_value, returns) 或全为 None（无持仓时）。
    """
    repo = get_repository()
    positions = repo.get_all_positions()
    if not positions:
        return None, None, None, None

    portfolio = []
    total_value = Decimal("0")
    for p in positions:
        mv = p.market_value if p.market_value else Decimal("0")
        portfolio.append({"code": p.code, "weight": mv})
        total_value += mv

    if total_value > 0:
        for item in portfolio:
            item["weight"] = item["weight"] / total_value
    else:
        total_value = Decimal("1000000")

    cash = Decimal("0")
    returns: list[Decimal] = []
    return portfolio, cash, total_value, returns


@router.get("/risk/status")
async def get_risk_status():
    """获取当前风控状态 — 基于真实持仓数据。"""
    portfolio, cash, total_value, returns = _build_portfolio_from_db()

    if not portfolio:
        # 无持仓时的兜底
        return {
            "risk_level": "LOW",
            "risk_score": 0.0,
            "blocked": False,
            "breaker_state": "CLOSED",
            "var_95": None,
            "advice": "暂无持仓数据",
        }

    rm = RiskManager()
    result = rm.check_all(portfolio, cash, total_value, returns)
    return {
        "risk_level": result["risk_level"],
        "risk_score": result["risk_score"],
        "blocked": result["blocked"],
        "breaker_state": result["breaker_state"],
        "var_95": (
            str(result["var_report"].get("var_95"))
            if result["var_report"].get("var_95")
            else None
        ),
        "advice": result["advice"],
    }


@ws_router.websocket("/ws/risk")
async def ws_risk(websocket: WebSocket):
    """风险监控实时推送 — 基于真实持仓和风控模块。"""
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
            portfolio, cash, total_value, returns = _build_portfolio_from_db()

            if portfolio:
                rm = RiskManager()
                result = rm.check_all(portfolio, cash, total_value, returns)
                payload = {
                    "type": "risk",
                    "risk_level": result["risk_level"],
                    "risk_score": result["risk_score"],
                    "breaker": result["breaker_state"],
                    "blocked": result["blocked"],
                    "var_95": str(result["var_report"].get("var_95")) if result["var_report"].get("var_95") else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                payload = {
                    "type": "risk",
                    "risk_level": "LOW",
                    "risk_score": 0.0,
                    "breaker": "CLOSED",
                    "blocked": False,
                    "var_95": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
