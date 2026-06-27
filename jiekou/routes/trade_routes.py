"""交易路由 — 完整选股→优化→风控→调仓→执行链路。"""
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Query
from sqlalchemy import text

from jiekou.dependencies import get_repository

router = APIRouter(prefix="/api", tags=["trade"])

# 初始资金（模拟账户）
DEFAULT_CAPITAL = Decimal("1000000")


@router.get("/trade/pipeline")
async def run_trade_pipeline(top_n: int = Query(default=10, ge=5, le=50)):
    """运行完整交易链路：选股 → 组合优化 → 风控 → 调仓 → 订单。

    Returns:
        { stocks, portfolio, risk, trades, orders, summary }
    """
    from juece.stock_selector import StockSelector
    from juece.portfolio_optimizer import PortfolioOptimizer
    from fengkong.risk_manager import RiskManager

    repo = get_repository()
    now = datetime.now(timezone.utc)

    # ── 1. 加载选股得分 ──
    with repo._session as s:
        latest = s.execute(text("SELECT MAX(trade_date) FROM fusion_score")).scalar()
        if not latest:
            return {"error": "no fusion_score data", "stocks": [], "trades": []}

        rows = s.execute(text(
            "SELECT code, composite_score FROM fusion_score WHERE trade_date=:d ORDER BY rank LIMIT :n"
        ), {"d": str(latest), "n": top_n}).fetchall()

        # 加载当前持仓
        pos_rows = []
        try:
            pos_rows = s.execute(text(
                "SELECT code, market_value, weight FROM position WHERE quantity > 0"
            )).fetchall()
        except Exception:
            pass

    stocks = [{"code": r[0], "score": round(float(r[1]), 4), "rank": i + 1}
              for i, r in enumerate(rows)]

    if not stocks:
        return {"error": "empty selection", "stocks": [], "trades": []}

    # ── 2. 选股信号 ──
    scores_dict = {s["code"]: Decimal(str(s["score"])) for s in stocks}
    selector = StockSelector(top_n=top_n)
    signals = selector.generate_signals(scores_dict)

    # ── 3. 组合优化 ──
    optimizer = PortfolioOptimizer(max_weight=Decimal("0.15"))
    picks_for_opt = [{"code": c, "score": Decimal(str(s["score"])), "weight": Decimal(str(1.0 / len(stocks)))}
                     for c, s in signals.items() if s["signal"] in ("BUY", "HOLD")]
    optimized = optimizer.optimize(picks_for_opt)

    # ── 4. 风控检查 ──
    portfolio_for_risk = [{"code": p["code"], "weight": Decimal(str(p["weight"]))}
                          for p in optimized]
    # 估算当前权益
    current_equity = DEFAULT_CAPITAL  # 简化：用初始资金
    risk_mgr = RiskManager()
    risk_result = risk_mgr.check_all(
        portfolio_for_risk,
        Decimal("0"), current_equity, [],
    )

    # ── 5. 调仓计算（组合优化结果 → 买卖订单）──
    current_positions = {}
    for r in pos_rows:
        code, mv, wt = r
        current_positions[code] = {
            "weight": float(wt) if wt else 0,
            "market_value": float(mv) if mv else 0,
        }

    # 直接根据优化权重生成买卖单（简化调仓——当前空仓时全部买入）
    buys = []
    sells = []
    cash_per_stock = float(current_equity) / max(len(optimized), 1)
    for p in optimized:
        buys.append({
            "code": p["code"], "action": "BUY",
            "weight": float(p["weight"]),
            "delta": float(p["weight"]),
            "amount": round(cash_per_stock * float(p["weight"]), 2),
            "reason": "组合优化配置",
        })

    # 持仓中有但不在优化组合中的 → 卖出
    for code in current_positions:
        if code not in {p["code"] for p in optimized}:
            sells.append({
                "code": code, "action": "SELL",
                "weight": current_positions[code]["weight"],
                "delta": -current_positions[code]["weight"],
                "amount": current_positions[code].get("market_value", 0),
                "reason": "调出组合",
            })

    return {
        "date": str(latest),
        "timestamp": now.isoformat(),
        "capital": float(current_equity),
        "stocks": stocks,
        "portfolio": [{"code": p["code"], "weight": float(p["weight"])} for p in optimized],
        "risk": {
            "level": risk_result.get("risk_level", "LOW"),
            "score": risk_result.get("risk_score", 0),
            "blocked": risk_result.get("blocked", False),
            "advice": risk_result.get("advice", ""),
        },
        "trades": {
            "buys": buys,
            "sells": sells,
            "buy_count": len(buys),
            "sell_count": len(sells),
            "net_cash_flow": sum(t["amount"] for t in sells) - sum(t["amount"] for t in buys),
        },
    }


@router.post("/trade/execute")
async def execute_trades():
    """执行当前调仓计划（模拟）。"""
    from zhixing.order_manager import OrderManager
    from zhixing.mock_broker import MockBroker
    from zhixing.trade_recorder import TradeRecorder

    # 先跑 pipeline 获取调仓计划
    pipeline = await run_trade_pipeline()
    if "error" in pipeline:
        return pipeline

    orders = []
    for t in pipeline["trades"]["buys"] + pipeline["trades"]["sells"]:
        orders.append({
            "code": t["code"],
            "direction": t["action"],
            "amount": t["amount"],
            "weight": t.get("weight", 0),
        })

    # 模拟执行
    broker = MockBroker()
    recorder = TradeRecorder()

    results = []
    for order in orders:
        result = broker.execute(order)
        recorder.record(result)
        results.append({
            "code": result["code"],
            "direction": result["direction"],
            "amount": float(result.get("filled_amount", 0)),
            "price": float(result.get("fill_price", 0)),
            "status": result.get("status", "filled"),
        })

    return {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "orders": results,
        "filled_count": len([r for r in results if r["status"] == "filled"]),
        "total_count": len(results),
    }


@router.get("/trade/history")
async def get_trade_history(limit: int = Query(default=20)):
    """最近的交易记录。"""
    repo = get_repository()
    try:
        with repo._session as s:
            rows = s.execute(text(
                "SELECT code, direction, filled_amount, fill_price, status, created_at "
                "FROM orders ORDER BY created_at DESC LIMIT :n"
            ), {"n": limit}).fetchall()
            return [
                {
                    "code": r[0], "direction": r[1],
                    "amount": float(r[2]) if r[2] else 0,
                    "price": float(r[3]) if r[3] else 0,
                    "status": r[4],
                    "time": str(r[5]) if r[5] else "",
                }
                for r in rows
            ]
    except Exception:
        return []
