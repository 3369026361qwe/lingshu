"""
赛道分析师 Agent — 行业比较和板块轮动分析。
"""

import time
from datetime import datetime
from decimal import Decimal

from zhinengti.agent_base import AgentBase, AgentOutput, AgentStatus
from zhinengti.prompt_templates import get_prompt


class SectorAnalyst(AgentBase):
    """赛道分析师 — 行业比较、板块轮动、赛道推荐。"""

    agent_id = "sector"
    name = "赛道分析师"
    description = "分析行业景气度、资金流向和板块轮动信号"

    def analyze(self, context: dict) -> AgentOutput:
        t0 = time.perf_counter()
        output = AgentOutput(agent_id=self.agent_id, timestamp=datetime.utcnow(), status=AgentStatus.RUNNING)

        try:
            prompt = self._build_prompt(context)
            response = self._call_llm(prompt)
            data = self._parse_response(response)

            confidence = Decimal(str(data.get("confidence", 0.5)))
            output.confidence = max(Decimal("0"), min(Decimal("1"), confidence))
            output.reasoning = data.get("reasoning", response[:200])
            output.evidence = data.get("evidence", [])
            output.target_stocks = self._extract_stocks(data)
            output.status = AgentStatus.DONE

        except Exception as exc:
            output.reasoning = f"赛道分析异常: {exc}"
            output.status = AgentStatus.ERROR

        output.latency_ms = int((time.perf_counter() - t0) * 1000)
        return output

    def _get_system_prompt(self) -> str:
        return get_prompt("sector")

    def _format_context(self, context: dict) -> str:
        parts = ["## 行业分析上下文"]
        industry_data = context.get("industry_data", {})
        if industry_data:
            for sector, info in industry_data.items():
                parts.append(f"{sector}: 资金流向={info.get('flow', 'N/A')}, 估值分位={info.get('pe_percentile', 'N/A')}")
        else:
            parts.append("(行业数据待接入)")
        sentiment = context.get("sentiment", {})
        if sentiment:
            parts.append(f"\n市场情绪: {sentiment.get('label', 'N/A')}")
        return "\n".join(parts)

    @staticmethod
    def _parse_response(response: str) -> dict:
        return AgentBase._parse_response(response, {"reasoning": response[:200]})

    @staticmethod
    def _extract_stocks(data: dict) -> list[str]:
        """从推荐赛道中提取相关股票（需结合行业映射）。"""
        sectors = data.get("top_sectors", [])
        # 占位：实际应从行业分类反查成分股
        return sectors[:5]
