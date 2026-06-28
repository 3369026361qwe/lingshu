"""
Agent 抽象基类 + AgentOutput 通信协议。

所有 5 个专业 Agent 继承此基类，输出统一的 AgentOutput 结构。
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from zhinengti.metrics import agent_analysis_duration, agent_analysis_total, llm_call_latency, llm_call_total


class AgentStatus(str, Enum):
    """Agent 运行状态。"""
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentOutput:
    """Agent 输出的结构化数据格式（通信协议）。

    所有 Agent 必须以此格式输出，供 Orchestrator 聚合和前端展示。
    """
    agent_id: str                                    # 智能体标识: macro/sector/stock/sentiment/risk
    timestamp: datetime                              # 分析时间
    target_stocks: list[str] = field(default_factory=list)  # 覆盖的股票列表
    signal: Decimal = field(default=Decimal("0"))    # [-1, 1] 看空→看多
    confidence: Decimal = field(default=Decimal("0")) # [0, 1] 置信度
    reasoning: str = ""                              # 推理过程（前端展示核心）
    evidence: list[dict] = field(default_factory=list)  # [{source, metric, value}]
    risk_flags: list[str] = field(default_factory=list)  # 识别的风险点
    raw_response: str = ""                           # LLM 原始响应（调试用）
    model_used: str = ""                             # 使用的模型
    tokens_used: int = 0                             # 消耗 Token
    latency_ms: int = 0                              # 响应延迟 (ms)
    status: AgentStatus = AgentStatus.IDLE            # 运行状态


class AgentBase(ABC):
    """Agent 抽象基类。

    子类必须:
        - 设置 agent_id, name, description
        - 实现 analyze(context) → AgentOutput
        - 可选覆盖 _build_prompt(context) 自定义提示词构建
    """

    agent_id: str = ""
    name: str = ""
    description: str = ""

    def __init__(self, llm_client=None, tools: dict | None = None):
        """
        Args:
            llm_client: LLM 客户端（默认从环境变量创建）
            tools: 工具字典 {tool_name: callable}
        """
        self._llm = llm_client
        self._tools = tools or {}

    @abstractmethod
    def analyze(self, context: dict) -> AgentOutput:
        """执行分析并返回结构化输出。

        Args:
            context: 分析上下文，包含市场数据、财务数据、因子值等

        Returns:
            AgentOutput 结构化分析结果
        """
        ...

    def _build_prompt(self, context: dict) -> str:
        """构建发送给 LLM 的提示词（子类可覆盖）。"""
        return self._get_system_prompt() + "\n\n" + self._format_context(context)

    def _get_system_prompt(self) -> str:
        """获取系统提示词（子类覆盖）。"""
        return f"你是{self.name}，{self.description}"

    def _format_context(self, context: dict) -> str:
        """格式化分析上下文为文本。"""
        return str(context)

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM（含 Prometheus 指标）。"""
        t0 = time.perf_counter()
        try:
            if self._llm:
                result = self._llm(prompt)
            else:
                from zhinengti.llm_client import get_llm_client
                client = get_llm_client()
                result = client(prompt)
            llm_call_total.labels(agent_id=self.agent_id, model="llm", status="success").inc()
            llm_call_latency.labels(agent_id=self.agent_id).observe(time.perf_counter() - t0)
            return result
        except Exception:
            llm_call_total.labels(agent_id=self.agent_id, model="llm", status="failure").inc()
            raise

    @staticmethod
    def _parse_response(response: str, defaults: dict | None = None) -> dict:
        """P2-4: 统一的 LLM JSON 响应解析（消除 5 处重复代码）。"""
        try:
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return defaults or {"reasoning": response[:200]}

    def _use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """调用注册的工具。"""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not registered for {self.agent_id}")
        return self._tools[tool_name](*args, **kwargs)

    def _record_analysis(self, latency_ms: int, status: str = "done") -> None:
        """记录 Agent 分析的 Prometheus 指标。"""
        agent_analysis_total.labels(agent_id=self.agent_id, status=status).inc()
        agent_analysis_duration.labels(agent_id=self.agent_id).observe(latency_ms / 1000.0)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id}>"
