"""
风险监控 Agent — 独立风险判断，有权否决交易。
"""

import time
from datetime import datetime
from decimal import Decimal

from zhinengti.agent_base import AgentBase, AgentOutput, AgentStatus
from zhinengti.prompt_templates import get_prompt


class RiskMonitor(AgentBase):
    """风险监控官 — 独立于收益导向Agent，有权否决交易。"""

    agent_id = "risk"
    name = "风险监控官"
    description = "独立风险评估，监控市场/持仓/流动性风险，有权否决交易"

    def analyze(self, context: dict) -> AgentOutput:
        t0 = time.perf_counter()
        output = AgentOutput(agent_id=self.agent_id, timestamp=datetime.utcnow(), status=AgentStatus.RUNNING)

        try:
            # 本地风险规则检查（不依赖 LLM）
            local_risk = self._local_risk_check(context)

            # LLM 深度风险分析
            prompt = self._build_prompt(context)
            response = self._call_llm(prompt)
            data = self._parse_response(response)

            # 合并：本地规则优先
            data.get("risk_level", local_risk["risk_level"])
            risk_score = max(int(data.get("risk_score", 1)), local_risk["risk_score"])
            veto = data.get("veto", False) or local_risk["veto"]

            output.signal = Decimal(str(-risk_score / 5))  # 风险越高信号越负
            output.confidence = Decimal(str(data.get("confidence", 0.8)))
            output.reasoning = data.get("reasoning", response[:200])
            output.evidence = data.get("evidence", [])
            output.risk_flags = data.get("risk_flags", []) + local_risk["risk_flags"]
            output.status = AgentStatus.DONE

            # 在 reasoning 中附加 veto 信息
            if veto:
                output.reasoning = f"[否决] {output.reasoning}"

        except Exception as exc:
            output.reasoning = f"风险分析异常: {exc}"
            output.status = AgentStatus.ERROR

        output.latency_ms = int((time.perf_counter() - t0) * 1000)
        return output

    def _get_system_prompt(self) -> str:
        return get_prompt("risk")

    def _format_context(self, context: dict) -> str:
        parts = ["## 风险分析上下文"]

        positions = context.get("positions", {})
        if positions:
            parts.append("\n### 当前持仓")
            for code, info in positions.items():
                parts.append(f"  {code}: 数量={info.get('quantity', 0)}, 权重={info.get('weight', 'N/A')}")

        market_data = context.get("market_data", {})
        if market_data:
            parts.append("\n### 市场波动")
            for code, data in list(market_data.items())[:5]:
                dates = sorted(data.keys())
                if len(dates) >= 2:
                    last_close = data[dates[-1]].get("close", 0)
                    prev_close = data[dates[-2]].get("close", 1)
                    change = (last_close - prev_close) / prev_close * 100 if prev_close else 0
                    parts.append(f"  {code}: 最新={last_close}, 日涨跌={change:.2f}%")

        sentiment = context.get("sentiment", {})
        if sentiment:
            parts.append(f"\n市场情绪: {sentiment.get('label', 'N/A')}")

        return "\n".join(parts)

    def _local_risk_check(self, context: dict) -> dict:
        """本地风险规则引擎（不依赖LLM，确定性检查）。"""
        risk_flags = []
        risk_score = 1
        veto = False

        positions = context.get("positions", {})
        # 规则1: 单票仓位超 10%
        for code, info in positions.items():
            weight = info.get("weight", 0)
            if weight and float(weight) > 0.10:
                risk_flags.append(f"单票仓位超限: {code} {weight}")
                risk_score = max(risk_score, 3)

        # 规则2: 行业集中度检查
        industry_weights = context.get("industry_weights", {})
        for ind, w in industry_weights.items():
            if float(w) > 0.30:
                risk_flags.append(f"行业集中度超限: {ind} {w}")
                risk_score = max(risk_score, 3)

        # 规则3: 总仓位超 95%
        total_weight = sum(float(info.get("weight", 0)) for info in positions.values())
        if total_weight > 0.95:
            risk_flags.append(f"总仓位超限: {total_weight:.1%}")
            risk_score = max(risk_score, 2)

        # 规则4: 情绪极端化检查
        sentiment = context.get("sentiment", {})
        sentiment_index = sentiment.get("index", 0)
        if abs(float(sentiment_index)) > 0.7:
            risk_flags.append(f"市场情绪极端化: {sentiment_index}")
            risk_score = max(risk_score, 3)

        risk_levels = {1: "LOW", 2: "GUARDED", 3: "ELEVATED", 4: "HIGH", 5: "CRITICAL"}
        return {
            "risk_level": risk_levels.get(risk_score, "LOW"),
            "risk_score": risk_score,
            "risk_flags": risk_flags,
            "veto": veto,
        }

    @staticmethod
    def _parse_response(response: str) -> dict:
        return AgentBase._parse_response(response, {"reasoning": response[:200], "risk_level": "GUARDED", "risk_score": 2})
