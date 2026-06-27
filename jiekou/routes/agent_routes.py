"""智能体路由 — Agent 报告查询 + Agent 实时 WebSocket。"""

from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from jiekou.dependencies import get_repository

router = APIRouter(prefix="/api", tags=["agents"])
ws_router = APIRouter(tags=["agents-ws"])


@router.get("/agents/reports")
async def get_agent_reports(agent_id: str = "", limit: int = 10):
    """获取最新的 Agent 分析报告。"""
    repo = get_repository()
    reports = repo.get_latest_agent_reports(agent_id=agent_id or None, limit=limit)
    return [
        {
            "agent_id": r.agent_id,
            "timestamp": r.analysis_date.isoformat(),
            "signal": str(r.signal),
            "confidence": str(r.confidence),
            "reasoning": r.reasoning[:500],
        }
        for r in reports
    ]


@ws_router.websocket("/ws/agents")
async def ws_agents(websocket: WebSocket):
    """Agent 实时分析推送 WebSocket — 从 DB 获取最新 Agent 报告。"""
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
            repo = get_repository()
            reports = repo.get_latest_agent_reports(limit=5)
            if reports:
                r = reports[0]  # 最新一条
                payload = {
                    "type": "agent",
                    "agent_id": r.agent_id,
                    "reasoning": r.reasoning[:200],
                    "confidence": str(r.confidence),
                    "signal": str(r.signal),
                    "timestamp": r.analysis_date.isoformat(),
                }
            else:
                payload = {
                    "type": "agent",
                    "agent_id": "system",
                    "reasoning": "暂无 Agent 分析报告，请等待日终分析或手动触发。",
                    "confidence": "0.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
