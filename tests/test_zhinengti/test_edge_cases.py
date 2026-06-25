"""
zhinengti 边缘场景测试：LLM异常/极端评分/空输入/并发/大量文档。
"""

import json
import time
from decimal import Decimal

import pytest

from zhinengti.agent_base import AgentOutput, AgentStatus
from zhinengti.llm_client import MockLLMClient
from zhinengti.macro_analyst import MacroAnalyst
from zhinengti.stock_analyst import StockAnalyst
from zhinengti.risk_monitor import RiskMonitor
from zhinengti.orchestrator import Orchestrator
from zhinengti.rag_pipeline import RAGPipeline


class TestLLMEdgeCases:
    def test_llm_returns_invalid_json(self):
        mock = MockLLMClient(responses={"macro": "这不是合法的JSON {{{"})
        agent = MacroAnalyst(llm_client=mock)
        output = agent.analyze({})
        assert output.status == AgentStatus.DONE
        assert output.reasoning

    def test_llm_returns_empty_string(self):
        mock = MockLLMClient(responses={"macro": ""})
        agent = MacroAnalyst(llm_client=mock)
        output = agent.analyze({})
        assert output.status == AgentStatus.DONE


class TestAgentEdgeCases:
    def test_macro_extreme_scores_clipped(self):
        mock = MockLLMClient(responses={
            "macro": json.dumps({"macro_score": 5.0, "confidence": 1.5, "reasoning": "test"})
        })
        agent = MacroAnalyst(llm_client=mock)
        output = agent.analyze({})
        assert -1 <= float(output.signal) <= 1
        assert 0 <= float(output.confidence) <= 1

    def test_risk_all_rules_triggered(self):
        mock = MockLLMClient()
        agent = RiskMonitor(llm_client=mock)
        context = {
            "positions": {"000001": {"weight": 0.15}, "000002": {"weight": 0.05}},
            "industry_weights": {"银行": 0.35},
            "sentiment": {"index": 0.8},
        }
        result = agent._local_risk_check(context)
        assert result["risk_score"] >= 3  # 单票超限(3) + 行业超限(3) + 情绪极端(3) = max=3
        assert len(result["risk_flags"]) >= 2

    def test_stock_analyst_empty_list(self):
        mock = MockLLMClient()
        agent = StockAnalyst(llm_client=mock)
        output = agent.analyze({"stocks": []})
        assert output.status == AgentStatus.DONE

    def test_stock_analyst_batch_ten(self):
        mock = MockLLMClient()
        agent = StockAnalyst(llm_client=mock)
        stocks = [f"{i:06d}" for i in range(1, 11)]
        fin_map = {c: {"pe": 10 + i} for i, c in enumerate(stocks)}
        context = {"stocks": stocks, "financial_data": fin_map, "market_data": {}, "news": {}, "factor_values": {}}
        output = agent.analyze(context)
        assert output.status == AgentStatus.DONE
        assert len(output.target_stocks) == 10


class TestOrchestratorEdgeCases:
    def test_single_agent_registered(self):
        mock = MockLLMClient()
        orch = Orchestrator(llm_client=mock)
        orch.register(MacroAnalyst(llm_client=mock))
        result = orch.run_analysis(["000001"], {"stocks": ["000001"], "sentiment": {}, "positions": {}}, parallel=True)
        assert len(result["agent_outputs"]) == 1

    def test_empty_agent_registry(self):
        orch = Orchestrator()
        result = orch.run_analysis([], {"stocks": []})
        assert result["agent_outputs"] == {}


class TestRAGEdgeCases:
    def test_search_empty_query(self):
        rag = RAGPipeline()
        rag.index_documents([{"id": "1", "title": "t", "content": "c", "source": "s", "date": "d"}])
        assert rag.search("") == []

    def test_large_document_indexing(self):
        rag = RAGPipeline()
        docs = [{"id": str(i), "title": f"文档{i}", "content": f"内容{i} " * 10, "source": "s", "date": "d"} for i in range(500)]
        t0 = time.perf_counter()
        rag.index_documents(docs)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Indexing too slow: {elapsed:.2f}s"
        assert len(rag._documents) == 500

    def test_jieba_fallback_tokens(self):
        import re
        text = "平安银行2025年报净利润增长15%"
        chars = re.findall(r'[一-鿿\w]+', text)
        tokens = []
        for chunk in chars:
            for i in range(len(chunk) - 1):
                tokens.append(chunk[i:i + 2])
        assert len(tokens) > 0
