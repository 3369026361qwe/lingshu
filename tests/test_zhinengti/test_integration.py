"""
zhinengti 集成测试：Agent → shujuku 持久化完整链路。
"""

from decimal import Decimal

import pytest

from shujuku.repository import Repository
from shujuku.session import SessionContext, init_db
from zhinengti.llm_client import MockLLMClient
from zhinengti.orchestrator import create_default_orchestrator


@pytest.fixture(autouse=True)
def setup_db():
    init_db(drop_all=True)


class TestAgentPersistence:
    def test_orchestrator_persists_reports(self):
        mock = MockLLMClient()
        orch = create_default_orchestrator(llm_client=mock)
        context = {
            "stocks": ["000001", "000002"],
            "sentiment": {"index": Decimal("0.1"), "label": "偏乐观"},
            "positions": {"000001": {"weight": 0.05}},
        }
        result = orch.run_analysis(["000001", "000002"], context, parallel=False)
        assert len(result["agent_outputs"]) == 5

        with SessionContext() as s:
            repo = Repository(s)
            reports = repo.get_latest_agent_reports(limit=10)
            assert len(reports) >= 1

    def test_multiple_runs_accumulate_reports(self):
        mock = MockLLMClient()
        orch = create_default_orchestrator(llm_client=mock)
        context = {"stocks": ["000001"], "sentiment": {}, "positions": {}}
        orch.run_analysis(["000001"], context, parallel=False)
        orch.run_analysis(["000001"], context, parallel=False)

        with SessionContext() as s:
            repo = Repository(s)
            reports = repo.get_latest_agent_reports(limit=20)
            assert len(reports) >= 8  # 2 runs × 5 agents, some may be error status


class TestAgentContextBuilding:
    def test_toolkit_builds_context(self):
        from zhinengti.agent_tools import AgentToolkit
        toolkit = AgentToolkit()
        context = toolkit.build_context(["000001"])
        assert "stocks" in context
        assert "sentiment" in context
        assert "market_data" in context
