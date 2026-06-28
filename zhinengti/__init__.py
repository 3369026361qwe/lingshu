"""
zhinengti — LLM 多智能体系统 ★ 核心创新

5 个专业 Agent + Orchestrator 调度 + RAG 检索增强 + 工具链集成。

Agent 体系:
    MacroAnalyst     — 宏观分析师 (经济周期/政策/系统性风险)
    SectorAnalyst    — 赛道分析师 (行业比较/板块轮动/资金流向)
    StockAnalyst     — 个股分析师 (财务分析/估值模型/RAG)
    SentimentAnalyst — 舆情分析师 (新闻情感/社交热度/情绪指数)
    RiskMonitor      — 风险监控官 (独立否决权/风控规则引擎)

Usage:
    from zhinengti import create_default_orchestrator
    orch = create_default_orchestrator()
    report = orch.run_daily_analysis(["000001", "000002"])
"""

from zhinengti.agent_base import AgentBase, AgentOutput, AgentStatus
from zhinengti.agent_tools import AgentToolkit
from zhinengti.macro_analyst import MacroAnalyst
from zhinengti.orchestrator import Orchestrator, create_default_orchestrator
from zhinengti.rag_pipeline import RAGPipeline
from zhinengti.risk_monitor import RiskMonitor
from zhinengti.sector_analyst import SectorAnalyst
from zhinengti.sentiment_analyst import SentimentAnalyst
from zhinengti.stock_analyst import StockAnalyst

__all__ = [
    "AgentBase", "AgentOutput", "AgentStatus",
    "MacroAnalyst", "SectorAnalyst", "StockAnalyst",
    "SentimentAnalyst", "RiskMonitor",
    "Orchestrator", "create_default_orchestrator",
    "RAGPipeline", "AgentToolkit",
]
