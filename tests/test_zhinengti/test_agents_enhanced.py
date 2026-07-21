"""
zhinengti/ 智能体模块补充测试 — 工具链/Prompt/AgentBase。
Run: python -m pytest tests/test_zhinengti/test_agents_enhanced.py -v
"""
from zhinengti.agent_tools import AgentToolkit


class TestAgentToolkit:
    def test_init(self):
        tk = AgentToolkit()
        assert tk is not None

    def test_query_industry_map(self):
        tk = AgentToolkit()
        result = tk.query_industry_map()
        assert isinstance(result, dict)

    def test_get_market_sentiment(self):
        tk = AgentToolkit()
        result = tk.get_market_sentiment()
        assert isinstance(result, dict)

    def test_build_context(self):
        tk = AgentToolkit()
        ctx = tk.build_context(["000001", "000002"])
        assert isinstance(ctx, dict)

    def test_query_factor_values(self):
        tk = AgentToolkit()
        fv = tk.query_factor_values("000001", ["roe", "pb"])
        assert isinstance(fv, dict)


class TestPromptTemplates:
    def test_macro_prompt(self):
        from zhinengti.prompt_templates import get_prompt
        p = get_prompt("macro")
        assert isinstance(p, str) and len(p) > 0

    def test_sector_prompt(self):
        from zhinengti.prompt_templates import get_prompt
        p = get_prompt("sector")
        assert isinstance(p, str) and len(p) > 0

    def test_stock_prompt(self):
        from zhinengti.prompt_templates import get_prompt
        p = get_prompt("stock")
        assert isinstance(p, str) and len(p) > 0

    def test_sentiment_prompt(self):
        from zhinengti.prompt_templates import get_prompt
        p = get_prompt("sentiment")
        assert isinstance(p, str) and len(p) > 0

    def test_risk_prompt(self):
        from zhinengti.prompt_templates import get_prompt
        p = get_prompt("risk")
        assert isinstance(p, str) and len(p) > 0

    def test_orchestrator_prompt(self):
        from zhinengti.prompt_templates import get_prompt
        p = get_prompt("orchestrator")
        assert isinstance(p, str) and len(p) > 0


class TestAgentBase:
    def test_agent_base_importable(self):
        """AgentBase 是可导入的抽象基类."""
        from zhinengti.agent_base import AgentBase
        assert AgentBase is not None

    def test_macro_analyst_importable(self):
        """具体 Agent 可导入."""
        from zhinengti.macro_analyst import MacroAnalyst
        assert MacroAnalyst is not None

    def test_sector_analyst_importable(self):
        from zhinengti.sector_analyst import SectorAnalyst
        assert SectorAnalyst is not None

    def test_risk_monitor_importable(self):
        from zhinengti.risk_monitor import RiskMonitor
        assert RiskMonitor is not None
