"""
RAG 检索增强生成管道。

为 Stock Analyst Agent 提供财报/公告/研报的向量检索能力。
当前版本使用关键词检索（无需外部 embedding 模型），后期可升级为向量检索。

Usage:
    rag = RAGPipeline()
    rag.index_documents(docs)
    results = rag.search("贵州茅台 2025年报 净利润", top_k=5)
"""

import json
import logging
import re
import time
from typing import Any, Optional

from zhinengti.metrics import rag_search_latency, rag_documents_indexed

_logger = logging.getLogger(__name__)


class RAGPipeline:
    """检索增强生成管道。

    当前使用 TF-IDF 风格的关键词匹配，后期可升级为:
        - sentence-transformers 向量化
        - Milvus/ChromaDB 向量存储
        - Hybrid Search (BM25 + Dense)
    """

    def __init__(self, cache_manager=None):
        self._documents: list[dict] = []
        self._index: dict[str, list[int]] = {}  # {word: [doc_indices]}
        self._cache = cache_manager

    # ── 索引 ────────────────────────────────────────────

    def index_documents(self, documents: list[dict]) -> int:
        """索引文档列表。

        Args:
            documents: [{"id": str, "title": str, "content": str, "source": str, "date": str}]

        Returns:
            索引的文档数
        """
        for doc in documents:
            doc_id = len(self._documents)
            self._documents.append(doc)

            # 分词 → 建立倒排索引
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            words = self._tokenize(text)
            for word in set(words):
                self._index.setdefault(word, []).append(doc_id)

        _logger.info("Indexed %d documents, %d unique words", len(self._documents), len(self._index))
        rag_documents_indexed.set(len(self._documents))
        return len(documents)

    # ── 检索 ────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """关键词检索。"""
        t0 = time.perf_counter()
        if not self._documents:
            rag_search_latency.observe(time.perf_counter() - t0)
            return []

        query_words = self._tokenize(query)
        if not query_words:
            return []

        # 计算每个文档的匹配分数
        scores: dict[int, float] = {}
        for word in query_words:
            if word not in self._index:
                continue
            idf = self._idf(word)
            for doc_id in self._index[word]:
                tf = self._tf(word, doc_id)
                scores[doc_id] = scores.get(doc_id, 0) + tf * idf

        # 排序取 Top-K
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result = [self._documents[doc_id] for doc_id, _ in ranked]
        rag_search_latency.observe(time.perf_counter() - t0)
        return result

    # ── 格式化 ───────────────────────────────────────────

    def format_context(self, query: str, top_k: int = 5) -> str:
        """检索并格式化为 LLM 可用的上下文。

        Returns:
            格式化的文本上下文
        """
        docs = self.search(query, top_k)
        if not docs:
            return "未找到相关文档。"
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"[文档{i}] {doc.get('title', '')}\n"
                f"来源: {doc.get('source', '')} | 日期: {doc.get('date', '')}\n"
                f"{doc.get('content', '')[:500]}\n"
            )
        return "\n---\n".join(parts)

    # ── 内部 ────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中文分词（基于 jieba，不可用时退化为字级别）。"""
        try:
            import jieba
            return [w.strip() for w in jieba.cut(text) if len(w.strip()) >= 2]
        except ImportError:
            # 退化为 2-gram 字符级分词
            chars = re.findall(r'[一-鿿\w]+', text)
            result = []
            for chunk in chars:
                for i in range(len(chunk) - 1):
                    result.append(chunk[i:i + 2])
            return result

    def _tf(self, word: str, doc_id: int) -> float:
        """词频 (Term Frequency)。"""
        text = f"{self._documents[doc_id].get('title', '')} {self._documents[doc_id].get('content', '')}"
        words = self._tokenize(text)
        if not words:
            return 0
        return words.count(word) / len(words)

    def _idf(self, word: str) -> float:
        """逆文档频率 (Inverse Document Frequency)。"""
        doc_count = len(self._index.get(word, []))
        if doc_count == 0:
            return 0
        import math
        return math.log((len(self._documents) + 1) / (doc_count + 1)) + 1

    # ── 工具 ────────────────────────────────────────────

    def index_from_news(self, news_items: list[dict]) -> int:
        """从新闻数据批量索引。"""
        docs = [
            {
                "id": item.get("id", f"news_{i}"),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "source": item.get("source", ""),
                "date": item.get("pub_time", ""),
            }
            for i, item in enumerate(news_items)
        ]
        return self.index_documents(docs)

    def clear(self) -> None:
        """清空索引。"""
        self._documents.clear()
        self._index.clear()
