"""
舆情分析师 Agent — 市场情绪监测和舆情信号提取。
"""

import time
from datetime import datetime
from decimal import Decimal

from zhinengti.agent_base import AgentBase, AgentOutput, AgentStatus
from zhinengti.prompt_templates import get_prompt


class SentimentAnalyst(AgentBase):
    """舆情分析师 — 新闻情感、社交热度、市场情绪指数。"""

    agent_id = "sentiment"
    name = "舆情分析师"
    description = "分析新闻情感、社交媒体热度和市场情绪信号"

    def analyze(self, context: dict) -> AgentOutput:
        t0 = time.perf_counter()
        output = AgentOutput(agent_id=self.agent_id, timestamp=datetime.utcnow(), status=AgentStatus.RUNNING)

        try:
            # 先从数据层获取情感评分（基于词典的快速分析）
            quick_sentiment = self._quick_sentiment_analysis(context)

            # 再用 LLM 进行深度情感分析
            prompt = self._build_prompt(context)
            response = self._call_llm(prompt)
            data = self._parse_response(response)

            # 融合快速分析和 LLM 分析
            sentiment_index = data.get("sentiment_index", quick_sentiment.get("index", 0))
            confidence = Decimal(str(data.get("confidence", 0.6)))

            output.signal = Decimal(str(sentiment_index))
            output.confidence = max(Decimal("0"), min(Decimal("1"), confidence))
            output.reasoning = data.get("reasoning", response[:200])
            output.evidence = data.get("evidence", [])
            output.risk_flags = data.get("risk_flags", [])
            output.target_stocks = [s.get("code", "") for s in data.get("stock_sentiments", [])]
            output.status = AgentStatus.DONE

        except Exception as exc:
            output.reasoning = f"舆情分析异常: {exc}"
            output.status = AgentStatus.ERROR

        output.latency_ms = int((time.perf_counter() - t0) * 1000)
        return output

    def _get_system_prompt(self) -> str:
        return get_prompt("sentiment")

    def _format_context(self, context: dict) -> str:
        parts = ["## 舆情分析上下文"]

        # 快速情感分析结果
        sentiment = context.get("sentiment", {})
        if sentiment:
            parts.append(f"市场情绪指数: {sentiment.get('index', 'N/A')} ({sentiment.get('label', '')})")
            parts.append(f"看多比例: {sentiment.get('bullish_ratio', 'N/A')}, 看空比例: {sentiment.get('bearish_ratio', 'N/A')}")

        # 个股舆情
        stock_sentiments = context.get("stock_sentiments", {})
        if stock_sentiments:
            parts.append("\n### 个股舆情")
            for code, sent in list(stock_sentiments.items())[:10]:
                parts.append(f"  {code}: 评分={sent.get('score', 0)}, 提及={sent.get('total_mentions', 0)}次")

        # 近期新闻标题
        news = context.get("news", {})
        if news:
            parts.append("\n### 近期新闻")
            for code, items in list(news.items())[:5]:
                for item in items[:3]:
                    parts.append(f"  {code}: {item.get('title', '')}")

        parts.append(f"\n分析时间: {datetime.utcnow().isoformat()}")
        return "\n".join(parts)

    def _quick_sentiment_analysis(self, context: dict) -> dict:
        """基于 shuju 层的情感词典快速分析。"""
        try:
            from shuju.sentiment_fetcher import SentimentFetcher
            fetcher = SentimentFetcher()
            return fetcher.get_market_sentiment_index()
        except Exception:
            return {"index": 0.0, "label": "无数据"}

    @staticmethod
    def _parse_response(response: str) -> dict:
        return AgentBase._parse_response(response, {"reasoning": response[:200], "sentiment_index": 0})
