"""
zhinengti 智能体系统 Prometheus 指标。
"""

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client import REGISTRY

# ── LLM 调用指标 ─────────────────────────────────────

llm_call_total = Counter(
    "lingshu_llm_call_total", "Total LLM API calls",
    ["agent_id", "model", "status"], registry=REGISTRY,
)

llm_call_latency = Histogram(
    "lingshu_llm_call_latency_seconds", "LLM API call latency",
    ["agent_id"], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0], registry=REGISTRY,
)

llm_tokens_total = Counter(
    "lingshu_llm_tokens_total", "Total tokens consumed",
    ["agent_id"], registry=REGISTRY,
)

# ── Agent 执行指标 ────────────────────────────────────

agent_analysis_total = Counter(
    "lingshu_agent_analysis_total", "Total agent analyses",
    ["agent_id", "status"], registry=REGISTRY,
)

agent_analysis_duration = Histogram(
    "lingshu_agent_analysis_duration_seconds", "Agent analysis duration",
    ["agent_id"], buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0], registry=REGISTRY,
)

# ── Orchestrator 指标 ─────────────────────────────────

orchestrator_runs_total = Counter(
    "lingshu_orchestrator_runs_total", "Total orchestrator analysis runs",
    ["mode"], registry=REGISTRY,
)

orchestrator_duration = Histogram(
    "lingshu_orchestrator_duration_seconds", "Orchestrator total analysis duration",
    ["mode"], buckets=[0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0], registry=REGISTRY,
)

# ── RAG 指标 ──────────────────────────────────────────

rag_search_latency = Histogram(
    "lingshu_rag_search_latency_seconds", "RAG search latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5], registry=REGISTRY,
)

rag_documents_indexed = Gauge(
    "lingshu_rag_documents_indexed", "Number of documents in RAG index",
    registry=REGISTRY,
)
