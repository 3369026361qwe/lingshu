"""选股路由 — Top-N 选股查询。"""
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from sqlalchemy import text

from jiekou.dependencies import get_repository

router = APIRouter(prefix="/api", tags=["selection"])


@router.get("/selection")
async def stock_selection(date: str = "", top_n: int = Query(default=30)):
    """选股接口 — 从 fusion_score 表读取最新选股。"""
    repo = get_repository()
    if not date:
        latest = repo.execute(text("SELECT MAX(trade_date) FROM fusion_score")).scalar()
        date = str(latest) if latest else ""
    rows = repo.execute(
        text("SELECT code, composite_score, rank FROM fusion_score WHERE trade_date = :d ORDER BY rank LIMIT :n"),
        {"d": date, "n": top_n},
    ).fetchall()
    picks = [{"code": r[0], "score": float(r[1]), "rank": r[2]} for r in rows]
    return {
        "date": date, "picks": picks, "count": len(picks),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
