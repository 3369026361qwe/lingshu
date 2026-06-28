"""
宏观分析师 Agent — 分析宏观经济环境并给出系统性评分。
"""

import time
from datetime import datetime
from decimal import Decimal

from zhinengti.agent_base import AgentBase, AgentOutput, AgentStatus
from zhinengti.prompt_templates import get_prompt


class MacroAnalyst(AgentBase):
    """宏观分析师 — 评估宏观环境和系统性风险。"""

    agent_id = "macro"
    name = "宏观分析师"
    description = "分析宏观经济指标，评估经济周期位置和系统性风险"

    def analyze(self, context: dict) -> AgentOutput:
        t0 = time.perf_counter()
        output = AgentOutput(agent_id=self.agent_id, timestamp=datetime.utcnow(), status=AgentStatus.RUNNING)

        try:
            # 构建提示词
            prompt = self._build_prompt(context)

            # 调用 LLM
            response = self._call_llm(prompt)

            # 解析结构化输出
            data = self._parse_response(response)
            macro_score = Decimal(str(data.get("macro_score", 0)))
            confidence = Decimal(str(data.get("confidence", 0.5)))

            output.signal = max(Decimal("-1"), min(Decimal("1"), macro_score))
            output.confidence = max(Decimal("0"), min(Decimal("1"), confidence))
            output.reasoning = data.get("reasoning", response[:200])
            output.evidence = data.get("evidence", [])
            output.risk_flags = data.get("risk_flags", [])
            output.raw_response = response
            output.model_used = "llm"
            output.status = AgentStatus.DONE

        except Exception as exc:
            output.reasoning = f"宏观分析异常: {exc}"
            output.status = AgentStatus.ERROR
            output.risk_flags = [str(exc)]

        output.latency_ms = int((time.perf_counter() - t0) * 1000)
        return output

    def _get_system_prompt(self) -> str:
        return get_prompt("macro")

    def _format_context(self, context: dict) -> str:
        """格式化宏观数据上下文。"""
        parts = ["## 当前宏观经济数据"]
        sentiment = context.get("sentiment", {})
        if sentiment:
            parts.append(f"市场情绪指数: {sentiment.get('index', 'N/A')} ({sentiment.get('label', '')})")

        # 从 context 提取宏观指标
        macro_indicators = context.get("macro_indicators", {})
        if macro_indicators:
            for name, value in macro_indicators.items():
                parts.append(f"{name}: {value}")
        else:
            parts.append("(宏观数据待接入)")

        parts.append(f"\n分析时间: {datetime.utcnow().isoformat()}")
        return "\n".join(parts)

    @staticmethod
    def _parse_response(response: str) -> dict:
        return AgentBase._parse_response(response, {"reasoning": response[:200], "macro_score": 0, "confidence": 0.5})
