"""
测试 Agent 基类、Orchestrator、5个Agent（使用 MockLLMClient）。
"""

import json
from datetime import datetime
from decimal import Decimal

import pytest

from zhinengti.agent_base import AgentBase, AgentOutput, AgentStatus
from zhinengti.llm_client import MockLLMClient
from zhinengti.macro_analyst import MacroAnalyst
from zhinengti.sector_analyst import SectorAnalyst
from zhinengti.stock_analyst import StockAnalyst
from zhinengti.sentiment_analyst import SentimentAnalyst
from zhinengti.risk_monitor import RiskMonitor
from zhinengti.orchestrator import Orchestrator, create_default_orchestrator
from zhinengti.rag_pipeline import RAGPipeline


@pytest.fixture
def mock_llm():
    return MockLLMClient()


class TestAgentOutput:
    def test_default_values(self):
        out = AgentOutput(agent_id="test", timestamp=datetime.utcnow())
        assert out.agent_id == "test"
        assert out.status == AgentStatus.IDLE
        assert out.signal == Decimal("0")
        assert out.confidence == Decimal("0")

    def test_custom_values(self):
        out = AgentOutput(
            agent_id="macro", timestamp=datetime.utcnow(),
            signal=Decimal("0.3"), confidence=Decimal("0.78"),
            reasoning="PMI扩张", risk_flags=["风险1"],
            status=AgentStatus.DONE,
        )
        assert out.signal == Decimal("0.3")
        assert len(out.risk_flags) == 1


class TestMacroAnalyst:
    def test_analyze_with_mock(self, mock_llm):
        agent = MacroAnalyst(llm_client=mock_llm)
        output = agent.analyze({"sentiment": {"index": 0.15, "label": "偏乐观"}})
        assert output.agent_id == "macro"
        assert output.status == AgentStatus.DONE
        assert output.confidence > 0

    def test_analyze_empty_context(self, mock_llm):
        agent = MacroAnalyst(llm_client=mock_llm)
        output = agent.analyze({})
        assert output.status == AgentStatus.DONE


class TestSectorAnalyst:
    def test_analyze(self, mock_llm):
        agent = SectorAnalyst(llm_client=mock_llm)
        output = agent.analyze({"industry_data": {"新能源": {"flow": "+52亿"}}})
        assert output.agent_id == "sector"
        assert output.status == AgentStatus.DONE


class TestStockAnalyst:
    def test_analyze(self, mock_llm):
        agent = StockAnalyst(llm_client=mock_llm)
        context = {
            "stocks": ["000001"],
            "financial_data": {"000001": {"pe": 15, "roe": 12}},
            "market_data": {"000001": {"20260601": {"close": 10.5}}},
            "news": {"000001": []},
            "factor_values": {},
        }
        output = agent.analyze(context)
        assert output.agent_id == "stock"
        assert output.status == AgentStatus.DONE

    def test_with_rag(self, mock_llm):
        rag = RAGPipeline()
        rag.index_documents([
            {"id": "1", "title": "平安银行2025年报", "content": "净利润增长15%", "source": "公告", "date": "2026-03-31"}
        ])
        agent = StockAnalyst(llm_client=mock_llm, rag_pipeline=rag)
        context = {"stocks": ["000001"], "financial_data": {}, "market_data": {}, "news": {}, "factor_values": {}}
        output = agent.analyze(context)
        assert output.status == AgentStatus.DONE


class TestSentimentAnalyst:
    def test_analyze(self, mock_llm):
        agent = SentimentAnalyst(llm_client=mock_llm)
        output = agent.analyze({"sentiment": {"index": 0.1, "label": "偏乐观"}})
        assert output.agent_id == "sentiment"
        assert output.status == AgentStatus.DONE


class TestRiskMonitor:
    def test_analyze(self, mock_llm):
        agent = RiskMonitor(llm_client=mock_llm)
        output = agent.analyze({"positions": {"000001": {"weight": 0.05}}})
        assert output.agent_id == "risk"

    def test_local_risk_check_position_limit(self, mock_llm):
        agent = RiskMonitor(llm_client=mock_llm)
        result = agent._local_risk_check({"positions": {"000001": {"weight": 0.15}}})
        assert result["risk_score"] >= 3
        assert any("单票" in f for f in result["risk_flags"])

    def test_local_risk_check_normal(self, mock_llm):
        agent = RiskMonitor(llm_client=mock_llm)
        result = agent._local_risk_check({"positions": {"000001": {"weight": 0.05}}})
        assert result["risk_score"] == 1


class TestOrchestrator:
    def test_create_default(self, mock_llm):
        orch = create_default_orchestrator(llm_client=mock_llm)
        assert len(orch._agents) == 5

    def test_run_analysis(self, mock_llm):
        orch = create_default_orchestrator(llm_client=mock_llm)
        context = {
            "stocks": ["000001"],
            "financial_data": {"000001": {"pe": 15}},
            "market_data": {"000001": {"20260601": {"close": 10.5}}},
            "news": {"000001": []},
            "sentiment": {"index": 0.1, "label": "偏乐观"},
            "factor_values": {"000001": {"pe": 15}},
            "positions": {"000001": {"weight": 0.05}},
        }
        report = orch.run_analysis(["000001"], context, parallel=True)
        assert "agent_outputs" in report
        assert "report" in report
        assert len(report["agent_outputs"]) == 5

    def test_run_sequential(self, mock_llm):
        orch = create_default_orchestrator(llm_client=mock_llm)
        context = {"stocks": ["000001"], "sentiment": {}, "positions": {}}
        report = orch.run_analysis(["000001"], context, parallel=False)
        assert len(report["agent_outputs"]) == 5


class TestRAGPipeline:
    def test_index_and_search(self):
        rag = RAGPipeline()
        rag.index_documents([
            {"id": "1", "title": "平安银行年报", "content": "净利润增长15%，ROE提升至13%", "source": "公告", "date": "2026-03-31"},
            {"id": "2", "title": "贵州茅台年报", "content": "营收增长12%，分红率提升", "source": "公告", "date": "2026-03-31"},
        ])
        results = rag.search("平安银行 年报 净利润", top_k=1)
        assert len(results) == 1
        assert "平安银行" in results[0]["title"]

    def test_empty_search(self):
        rag = RAGPipeline()
        assert rag.search("test") == []

    def test_format_context(self):
        rag = RAGPipeline()
        rag.index_documents([
            {"id": "1", "title": "测试公告", "content": "这是一个测试。", "source": "测试", "date": "2026-01-01"}
        ])
        ctx = rag.format_context("测试", top_k=1)
        assert "测试公告" in ctx

    def test_clear(self):
        rag = RAGPipeline()
        rag.index_documents([{"id": "1", "title": "t", "content": "c", "source": "s", "date": "d"}])
        rag.clear()
        assert rag.search("t") == []
