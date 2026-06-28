"""
Orchestrator 调度协调器 — 任务分配、并行调度、结果聚合、报告生成。

核心流程:
    1. 接收分析任务 (全市场 / 指定股票列表)
    2. 构建分析上下文 (通过 AgentToolkit)
    3. 并行分派给 5 个 Agent
    4. 收集 Agent 输出
    5. 加权融合 → 生成最终投资报告

Usage:
    orch = Orchestrator()
    report = orch.run_daily_analysis(stock_list, context)
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal

from zhinengti.agent_base import AgentBase, AgentOutput, AgentStatus
from zhinengti.metrics import orchestrator_duration, orchestrator_runs_total

_logger = logging.getLogger(__name__)


class Orchestrator:
    """总调度官 — 协调 5 个专业 Agent 完成全市场分析。"""

    def __init__(self, llm_client=None, toolkit=None):
        self._llm = llm_client
        self._toolkit = toolkit
        self._agents: dict[str, AgentBase] = {}

    def register(self, agent: AgentBase) -> None:
        """注册 Agent。"""
        self._agents[agent.agent_id] = agent

    def register_all(self, agents: list[AgentBase]) -> None:
        """批量注册。"""
        for a in agents:
            self.register(a)

    # ── 核心流程 ────────────────────────────────────────

    def run_analysis(
        self,
        stock_list: list[str],
        context: dict | None = None,
        parallel: bool = True,
    ) -> dict:
        """执行完整分析流程。

        Args:
            stock_list: 待分析股票列表
            context: 预构建的上下文（None 则自动构建）
            parallel: 是否并行执行 Agent

        Returns:
            {agent_outputs: {agent_id: AgentOutput}, report: str, timestamp: datetime}
        """
        mode = "parallel" if parallel else "sequential"
        orchestrator_runs_total.labels(mode=mode).inc()
        t0 = time.perf_counter()

        # 构建上下文
        if context is None and self._toolkit:
            context = self._toolkit.build_context(stock_list)
        elif context is None:
            context = {"stocks": stock_list}

        # 并行执行所有 Agent
        if parallel and len(self._agents) > 1:
            agent_outputs = self._run_parallel(context)
        else:
            agent_outputs = self._run_sequential(context)

        # 生成综合报告
        report = self._generate_report(agent_outputs, context)

        elapsed = time.perf_counter() - t0
        orchestrator_duration.labels(mode=mode).observe(elapsed)
        _logger.info("Analysis completed in %.1fs for %d stocks", elapsed, len(stock_list))

        result = {
            "agent_outputs": agent_outputs,
            "report": report,
            "timestamp": datetime.utcnow(),
            "elapsed_seconds": round(elapsed, 1),
            "stocks_analyzed": len(stock_list),
        }
        # 持久化 Agent 报告
        self._persist_reports(result)
        return result

    def run_daily_analysis(self, stock_list: list[str]) -> dict:
        """每日盘后全市场分析（默认并行 + 自动构建上下文）。"""
        context = self._toolkit.build_context(stock_list) if self._toolkit else {"stocks": stock_list}
        return self.run_analysis(stock_list, context, parallel=True)

    def _persist_reports(self, result: dict) -> None:
        """将 Agent 输出持久化到数据库。"""
        try:
            from shujuku.repository import Repository
            from shujuku.session import SessionContext

            with SessionContext() as s:
                repo = Repository(s)
                for agent_id, output in result.get("agent_outputs", {}).items():
                    if output.status.value == "done":
                        repo.save_agent_report(
                            agent_id=agent_id,
                            analysis_date=output.timestamp,
                            target_stocks=json.dumps(output.target_stocks, ensure_ascii=False),
                            signal=output.signal,
                            confidence=output.confidence,
                            reasoning=output.reasoning[:1000],
                            risk_flags=json.dumps(output.risk_flags, ensure_ascii=False) if output.risk_flags else None,
                            model_used=output.model_used or "llm",
                            tokens_used=output.tokens_used,
                            latency_ms=output.latency_ms,
                        )
                s.commit()
            _logger.info("Persisted %d agent reports", len(result.get("agent_outputs", {})))
        except Exception as exc:
            _logger.warning("Failed to persist agent reports: %s", exc)

    # ── 内部 ────────────────────────────────────────────

    def _run_parallel(self, context: dict) -> dict[str, AgentOutput]:
        """并行执行所有 Agent。"""
        outputs = {}
        with ThreadPoolExecutor(max_workers=len(self._agents)) as executor:
            futures = {
                executor.submit(agent.analyze, context): agent_id
                for agent_id, agent in self._agents.items()
            }
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    outputs[agent_id] = future.result()
                except Exception as exc:
                    _logger.error("Agent %s failed: %s", agent_id, exc)
                    outputs[agent_id] = AgentOutput(
                        agent_id=agent_id,
                        timestamp=datetime.utcnow(),
                        status=AgentStatus.ERROR,
                        reasoning=str(exc),
                    )
        return outputs

    def _run_sequential(self, context: dict) -> dict[str, AgentOutput]:
        """串行执行（调试用）。"""
        outputs = {}
        for agent_id, agent in self._agents.items():
            try:
                outputs[agent_id] = agent.analyze(context)
            except Exception as exc:
                outputs[agent_id] = AgentOutput(agent_id=agent_id, timestamp=datetime.utcnow(), status=AgentStatus.ERROR, reasoning=str(exc))
        return outputs

    # ── 报告生成 ────────────────────────────────────────

    def _generate_report(self, outputs: dict[str, AgentOutput], context: dict) -> str:
        """综合各 Agent 输出，生成最终投资报告。"""
        parts = ["# 灵枢量化系统 — 每日投资报告", f"生成时间: {datetime.utcnow().isoformat()}\n"]

        # 宏观
        macro = outputs.get("macro")
        if macro and macro.status == AgentStatus.DONE:
            parts.append(f"## 宏观环境\n{macro.reasoning}\n置信度: {macro.confidence}")

        # 赛道
        sector = outputs.get("sector")
        if sector and sector.status == AgentStatus.DONE:
            parts.append(f"## 赛道推荐\n{sector.reasoning}")

        # 个股
        stock = outputs.get("stock")
        if stock and stock.status == AgentStatus.DONE:
            parts.append(f"## 个股分析\n{stock.reasoning}")

        # 舆情
        sentiment = outputs.get("sentiment")
        if sentiment and sentiment.status == AgentStatus.DONE:
            parts.append(f"## 市场情绪\n{sentiment.reasoning}")

        # 风险
        risk = outputs.get("risk")
        if risk and risk.status == AgentStatus.DONE:
            parts.append(f"## ⚠ 风险评估\n{risk.reasoning}")
            if risk.risk_flags:
                parts.append("### 风险提示")
                for flag in risk.risk_flags:
                    parts.append(f"  - {flag}")

        # 综合评分（加权融合）
        parts.append(self._compute_composite_score(outputs))

        return "\n\n".join(parts)

    def _compute_composite_score(self, outputs: dict[str, AgentOutput]) -> str:
        """加权融合: 0.25×宏观 + 0.25×赛道 + 0.30×个股 + 0.20×情绪。"""
        weights = {"macro": 0.25, "sector": 0.25, "stock": 0.30, "sentiment": 0.20}
        composite = Decimal("0")
        total_weight = Decimal("0")

        for agent_id, weight in weights.items():
            out = outputs.get(agent_id)
            if out and out.status == AgentStatus.DONE:
                composite += Decimal(str(weight)) * out.signal
                total_weight += Decimal(str(weight))

        if total_weight > 0:
            composite /= total_weight

        label = "强烈看多" if composite > 0.3 else ("偏积极" if composite > 0.1 else ("中性" if composite > -0.1 else ("偏消极" if composite > -0.3 else "强烈看空")))

        return f"\n## 综合评分\n信号: {float(composite):.3f} ({label})"


# ── 工厂函数 ──────────────────────────────────────────

def create_default_orchestrator(llm_client=None, toolkit=None) -> Orchestrator:
    """创建包含全部 5 个 Agent 的预配置调度器。"""
    from zhinengti.macro_analyst import MacroAnalyst
    from zhinengti.risk_monitor import RiskMonitor
    from zhinengti.sector_analyst import SectorAnalyst
    from zhinengti.sentiment_analyst import SentimentAnalyst
    from zhinengti.stock_analyst import StockAnalyst

    orch = Orchestrator(llm_client=llm_client, toolkit=toolkit)
    orch.register_all([
        MacroAnalyst(llm_client=llm_client),
        SectorAnalyst(llm_client=llm_client),
        StockAnalyst(llm_client=llm_client),
        SentimentAnalyst(llm_client=llm_client),
        RiskMonitor(llm_client=llm_client),
    ])
    return orch
