"""
LLM 调用客户端。

支持:
    - 本地 Qwen3-32B (通过 vLLM / Ollama 兼容 API)
    - 云端 API (deepseek-v4 / OpenAI 兼容)
    - Mock 模式 (测试用)

环境变量:
    LLM_BACKEND   — deepseek | qwen | openai | mock
    LLM_API_KEY   — API Key
    LLM_BASE_URL  — API 地址
    LLM_MODEL     — 模型名
"""

import json
import logging
import os
import re
import time
from typing import Any, Optional, Protocol

_logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """LLM 客户端协议。"""
    def __call__(self, prompt: str, **kwargs) -> str:
        ...


class MockLLMClient:
    """Mock 客户端：返回预设响应（测试用）。"""

    def __init__(self, responses: Optional[dict[str, str]] = None):
        self._responses = responses or {}
        self._call_count = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        self._call_count += 1
        # 尝试从 prompt 中提取 JSON 格式请求并返回 mock 数据
        for key, response in self._responses.items():
            if key in prompt:
                return response
        # 默认返回 mock JSON
        return json.dumps({
            "macro_score": 0.3,
            "confidence": 0.78,
            "reasoning": "Mock: PMI连续3月扩张，建议超配制造业",
            "evidence": [{"source": "Mock", "metric": "PMI", "value": "51.2"}],
            "risk_flags": [],
        }, ensure_ascii=False)

    @property
    def call_count(self) -> int:
        return self._call_count


class OpenAICompatibleClient:
    """OpenAI 兼容 API 客户端 (deepseek / Qwen / 通用)。"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        timeout: int = 60,
    ):
        self._api_key = api_key or os.getenv("LLM_API_KEY", "")
        self._base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self._model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self._timeout = timeout

    def __call__(self, prompt: str, **kwargs) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "你是一位专业的A股投资分析师。请严格以JSON格式输出，确保JSON有效可解析。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # 低温度，提高一致性
                max_tokens=2000,
            )

            content = response.choices[0].message.content or ""
            return self._extract_json(content)

        except Exception as exc:
            _logger.error("LLM call failed: %s", exc)
            return json.dumps({"error": str(exc), "reasoning": "LLM调用失败"}, ensure_ascii=False)

    @staticmethod
    def _extract_json(text: str) -> str:
        """从 LLM 响应中提取 JSON 部分。"""
        # 尝试直接解析
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 代码块
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            return match.group(1).strip()
        # 尝试找到 { 到 } 的范围
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text


class LocalQwenClient:
    """本地 Qwen 模型客户端 (vLLM / Ollama API)。"""

    def __init__(self, base_url: str = "", model: str = "", timeout: int = 120):
        self._base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self._model = model or os.getenv("LLM_MODEL", "qwen3:32b")
        self._timeout = timeout

    def __call__(self, prompt: str, **kwargs) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=self._base_url, api_key="not-needed", timeout=self._timeout)
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            content = response.choices[0].message.content or ""
            return OpenAICompatibleClient._extract_json(content)
        except Exception as exc:
            _logger.error("Local Qwen call failed: %s", exc)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)


# ── 工厂函数 ──────────────────────────────────────────

_client: Optional[LLMClient] = None


def get_llm_client(backend: str = "") -> LLMClient:
    """获取 LLM 客户端单例。"""
    global _client
    if _client is not None:
        return _client

    backend = backend or os.getenv("LLM_BACKEND", "mock")

    if backend == "mock":
        _client = MockLLMClient()
    elif backend == "qwen":
        _client = LocalQwenClient()
    elif backend in ("deepseek", "openai"):
        _client = OpenAICompatibleClient()
    else:
        _logger.warning("Unknown LLM backend '%s', using mock", backend)
        _client = MockLLMClient()

    return _client


def set_llm_client(client: LLMClient) -> None:
    """注入自定义 LLM 客户端（测试用）。"""
    global _client
    _client = client
