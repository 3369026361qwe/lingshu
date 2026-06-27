"""持仓管理路由 — 持仓查询 / 权益曲线 / 因子权重。"""

from fastapi import APIRouter
from sqlalchemy import text

from jiekou.dependencies import get_repository

router = APIRouter(prefix="/api", tags=["portfolio"])


@router.get("/portfolio")
async def get_portfolio():
    """当前持仓列表。"""
    repo = get_repository()
    positions = repo.get_all_positions()
    return [
        {
            "code": p.code,
            "quantity": p.quantity,
            "avg_cost": str(p.avg_cost),
            "market_value": str(p.market_value) if p.market_value else None,
            "weight": str(p.weight) if p.weight else None,
        }
        for p in positions
    ]


@router.get("/equity")
async def get_equity_curve():
    """日频权益曲线 — 供前端 Dashboard/回测绘制。"""
    repo = get_repository()
    with repo._session as s:
        rows = s.execute(
            text("SELECT trade_date, total_value FROM portfolio_snapshot ORDER BY trade_date")
        ).fetchall()
        return {
            "data": [
                {"date": str(r[0]), "value": r[1]}
                for r in rows
                if r[1] is not None
            ]
        }


@router.get("/factors/weights")
async def get_factor_weights():
    """因子权重列表 — 供前端 Dashboard 因子权重卡片展示。"""
    repo = get_repository()
    with repo._session as s:
        rows = s.execute(
            text("SELECT factor_name, weight FROM factor_weight ORDER BY weight DESC")
        ).fetchall()
        if not rows:
            return {
                "weights": [
                    {"name": "GNN", "weight": 0.22},
                    {"name": "Agent", "weight": 0.25},
                    {"name": "ROE", "weight": 0.18},
                    {"name": "PE", "weight": 0.15},
                    {"name": "动量", "weight": 0.12},
                    {"name": "情绪", "weight": 0.08},
                ]
            }
        return {"weights": [{"name": r[0], "weight": r[1]} for r in rows]}
