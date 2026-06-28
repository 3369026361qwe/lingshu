"""
个股分析师 Agent — 深度基本面分析和估值模型。
"""

import time
from datetime import datetime
from decimal import Decimal

from zhinengti.agent_base import AgentBase, AgentOutput, AgentStatus
from zhinengti.prompt_templates import get_prompt


class StockAnalyst(AgentBase):
    """个股分析师 — 深度财务分析、估值建模、质量评估。"""

    agent_id = "stock"
    name = "个股分析师"
    description = "分析个股财务报表、估值水平和盈利质量"

    def __init__(self, llm_client=None, tools=None, rag_pipeline=None):
        super().__init__(llm_client, tools)
        self._rag = rag_pipeline

    def analyze(self, context: dict) -> AgentOutput:
        t0 = time.perf_counter()
        output = AgentOutput(agent_id=self.agent_id, timestamp=datetime.utcnow(), status=AgentStatus.RUNNING)

        try:
            stock_list = context.get("stocks", [])[:10]
            output.target_stocks = stock_list

            if not stock_list:
                output.status = AgentStatus.DONE
                output.reasoning = "无待分析股票"
                output.latency_ms = int((time.perf_counter() - t0) * 1000)
                return output

            # P1-3: 批量分析 — 一次 LLM 调用处理多只股票
            multi_context = {
                "stocks": [self._build_stock_context(code, context) for code in stock_list]
            }
            prompt = self._build_batch_prompt(multi_context)
            response = self._call_llm(prompt)
            data = self._parse_response(response)

            analyses = data.get("stock_analysis", [])
            # 确保每只股票都有分析结果
            for i, code in enumerate(stock_list):
                if i < len(analyses):
                    analyses[i]["code"] = code

            output.reasoning = self._aggregate(analyses)
            output.evidence = self._collect_evidence(analyses)
            output.risk_flags = self._collect_risks(analyses)
            output.confidence = self._avg_confidence(analyses)
            output.status = AgentStatus.DONE

        except Exception as exc:
            output.reasoning = f"个股分析异常: {exc}"
            output.status = AgentStatus.ERROR

        output.latency_ms = int((time.perf_counter() - t0) * 1000)
        return output

    def _build_batch_prompt(self, context: dict) -> str:
        """构建批量分析提示词（一次分析多只股票）。"""
        system = self._get_system_prompt()
        stocks_text = []
        for stock_ctx in context.get("stocks", []):
            stocks_text.append(self._format_context(stock_ctx))
        return system + "\n\n" + "\n---\n".join(stocks_text) + "\n\n请以JSON格式输出全部股票的分析结果，格式为{\"stock_analysis\": [...]}"

    def _get_system_prompt(self) -> str:
        return get_prompt("stock")

    def _build_stock_context(self, code: str, context: dict) -> dict:
        """为单只股票构建分析上下文。"""
        fin = context.get("financial_data", {}).get(code, {})
        market = context.get("market_data", {}).get(code, {})
        factors = context.get("factor_values", {}).get(code, {})
        news = context.get("news", {}).get(code, [])

        # RAG 检索相关文档
        rag_context = ""
        if self._rag:
            company_name = fin.get("name", code)
            rag_context = self._rag.format_context(f"{company_name} 财报 公告 研报")

        return {
            "code": code,
            "financial": fin,
            "market_summary": f"最新收盘价: {list(market.values())[-1].get('close', 'N/A') if market else 'N/A'}",
            "factors": factors,
            "news_titles": [n.get("title", "") for n in news[:5]],
            "rag_context": rag_context,
        }

    def _format_context(self, context: dict) -> str:
        fin = context.get("financial", {})
        factors = context.get("factors", {})
        news = context.get("news_titles", [])
        rag = context.get("rag_context", "")

        parts = [
            f"## 股票: {context.get('code', '')}",
            f"市场数据: {context.get('market_summary', 'N/A')}",
            "\n### 财务数据",
        ]
        for k, v in fin.items():
            parts.append(f"  {k}: {v}")
        parts.append("\n### 因子值")
        for k, v in factors.items():
            parts.append(f"  {k}: {v}")
        if news:
            parts.append("\n### 近期新闻")
            for t in news:
                parts.append(f"  - {t}")
        if rag:
            parts.append(f"\n### 相关文档\n{rag[:1000]}")
        return "\n".join(parts)

    def _aggregate(self, analyses: list[dict]) -> str:
        if not analyses:
            return "无有效分析结果"
        lines = [f"{a.get('code', '?')}: {a.get('reasoning', a.get('strength', 'N/A'))}"[:200] for a in analyses]
        return "\n".join(lines)

    def _collect_evidence(self, analyses: list[dict]) -> list[dict]:
        evidence = []
        for a in analyses:
            evidence.extend(a.get("evidence", []))
        return evidence[:10]

    def _collect_risks(self, analyses: list[dict]) -> list[str]:
        risks = []
        for a in analyses:
            risks.extend(a.get("risk_flags", []))
        return list(set(risks))[:10]

    def _avg_confidence(self, analyses: list[dict]) -> Decimal:
        if not analyses:
            return Decimal("0")
        confs = [Decimal(str(a.get("confidence", 0.5))) for a in analyses]
        return sum(confs) / len(confs)

    @staticmethod
    def _parse_response(response: str) -> dict:
        return AgentBase._parse_response(response, {"reasoning": response[:200]})
