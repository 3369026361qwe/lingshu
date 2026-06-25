"""
社交媒体舆情获取器。

功能:
    - 微博/雪球热帖采集
    - 个股情感评分 (正面/负面/中性)
    - 市场情绪指数计算

特性:
    - 基于文本关键词 + 简单情感词典 (不依赖外部 LLM)
    - 可扩展为接入 NLP 模型
    - 优雅降级

Usage:
    fetcher = SentimentFetcher()
    score = fetcher.get_stock_sentiment("000001")
    index = fetcher.get_market_sentiment_index()
"""

import logging
import re
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from shuju.cache_manager import DataCacheManager

_logger = logging.getLogger(__name__)


# ── 简易中文金融情感词典 ──────────────────────────────

_POSITIVE_WORDS = {
    "涨停", "大涨", "上涨", "利好", "超预期", "增长", "突破", "创新高",
    "买入", "增持", "回购", "分红", "盈利", "扭亏", "龙头",
    "景气", "扩张", "中标", "获批", "量产", "订单", "放量",
    "反弹", "企稳", "修复", "低估", "优质", "升级",
}

_NEGATIVE_WORDS = {
    "跌停", "大跌", "下跌", "利空", "低于预期", "下滑", "跌破", "创新低",
    "卖出", "减持", "亏损", "暴雷", "退市", "监管", "处罚",
    "违约", "破产", "停产", "裁员", "诉讼", "冻结",
    "衰退", "恶化", "滞涨", "高估", "踩踏", "恐慌",
}

# 否定词：出现在情感词前 N 个字内则反转极性
_NEGATION_WORDS = {"不", "没", "无", "未", "非", "别", "莫", "勿", "否", "难以", "无法"}

_NEGATION_WINDOW = 5  # 否定词有效距离（字符数）

_NEUTRALIZING_PATTERNS = [
    re.compile(r"符合预期"),
    re.compile(r"低于市场传闻"),
    re.compile(r"但不改.*趋势"),
    re.compile(r"长期.*看好"),
]


class SentimentFetcher:
    """社交媒体舆情获取器。"""

    def __init__(self, cache: Optional[DataCacheManager] = None, news_fetcher=None) -> None:
        self._cache = cache or DataCacheManager()
        self._news_fetcher = news_fetcher  # 依赖注入，避免重复创建
        self._last_request = 0.0

    def _get_news_fetcher(self):
        """懒加载 NewsFetcher。"""
        if self._news_fetcher is None:
            from shuju.news_fetcher import NewsFetcher
            self._news_fetcher = NewsFetcher(cache=self._cache)
        return self._news_fetcher

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < 0.3:
            time.sleep(0.3 - elapsed)
        self._last_request = time.time()

    # ── 个股情感评分 ────────────────────────────────────

    def get_stock_sentiment(self, code: str) -> dict:
        """获取单只股票的情感评分。

        Returns:
            {"code": "000001", "score": 0.35, "positive_count": 12,
             "negative_count": 5, "neutral_count": 20, "total_mentions": 37,
             "sentiment_label": "偏积极"}
        """
        cached = self._cache.get_sentiment(code)
        if cached:
            return cached

        # 尝试从新闻标题中提取情感
        try:
            news_fetcher = self._get_news_fetcher()
            news_items = news_fetcher.get_stock_news(code, limit=30) if news_fetcher else []
        except Exception:
            news_items = []

        if not news_items:
            return self._empty_sentiment(code)

        positive = 0
        negative = 0
        neutral = 0

        for item in news_items:
            title = item.get("title", "")
            content = item.get("content", "")
            text = f"{title} {content}"

            pos_count, neg_count = self._count_sentiment_words(text)

            # 中性化模式检测
            neutralized = any(p.search(text) for p in _NEUTRALIZING_PATTERNS)

            if neutralized:
                neutral += 1
            elif pos_count > neg_count:
                positive += 1
            elif neg_count > pos_count:
                negative += 1
            else:
                neutral += 1

        total = positive + negative + neutral
        if total == 0:
            return self._empty_sentiment(code)

        score = (Decimal(str(positive - negative)) / Decimal(str(total))).quantize(Decimal("0.01"))

        result = {
            "code": code,
            "score": score,  # 保持 Decimal 精度
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "total_mentions": total,
            "sentiment_label": self._label(score),
        }

        self._cache.cache_sentiment(code, result)
        return result

    # ── 市场情绪指数 ────────────────────────────────────

    def get_market_sentiment_index(self, sample_codes: Optional[list[str]] = None) -> dict:
        """计算全市场情绪指数。

        Args:
            sample_codes: 采样股票代码列表，默认沪深300成分股代表性样本

        Returns:
            {"index": 0.15, "bullish_ratio": 0.45, "bearish_ratio": 0.30,
             "sample_size": 50, "label": "偏乐观"}
        """
        if sample_codes is None:
            # 默认使用代表性样本
            sample_codes = [
                "000001", "000002", "000858", "002415", "002594",
                "300750", "600036", "600519", "601012", "601318",
            ]

        scores = []
        for code in sample_codes:
            result = self.get_stock_sentiment(code)
            if result.get("total_mentions", 0) > 0:
                scores.append(result["score"])

        if not scores:
            return {"index": 0.0, "bullish_ratio": 0.0, "bearish_ratio": 0.0, "sample_size": 0, "label": "无数据"}

        avg_score = sum(scores) / len(scores)
        bullish = sum(1 for s in scores if s > 0.1)
        bearish = sum(1 for s in scores if s < -0.1)

        return {
            "index": round(avg_score, 2),
            "bullish_ratio": round(bullish / len(scores), 2),
            "bearish_ratio": round(bearish / len(scores), 2),
            "sample_size": len(scores),
            "label": self._market_label(avg_score),
        }

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _count_sentiment_words(text: str) -> tuple[int, int]:
        """统计正面/负面情感词出现次数，考虑否定词反转。支持多次匹配。

        否定词 + 情感词 → 反转极性（如"不好"不算正面，"没有下跌"不算负面）。
        """
        pos = 0
        neg = 0

        for word in _POSITIVE_WORDS:
            idx = 0
            while True:
                idx = text.find(word, idx)
                if idx == -1:
                    break
                prefix = text[max(0, idx - _NEGATION_WINDOW):idx]
                if any(nw in prefix for nw in _NEGATION_WORDS):
                    neg += 1
                else:
                    pos += 1
                idx += 1  # P2-7: 继续搜索下一个匹配

        for word in _NEGATIVE_WORDS:
            idx = 0
            while True:
                idx = text.find(word, idx)
                if idx == -1:
                    break
                prefix = text[max(0, idx - _NEGATION_WINDOW):idx]
                if any(nw in prefix for nw in _NEGATION_WORDS):
                    pos += 1
                else:
                    neg += 1
                idx += 1

        return pos, neg

    @staticmethod
    def _empty_sentiment(code: str) -> dict:
        return {
            "code": code, "score": 0.0,
            "positive_count": 0, "negative_count": 0, "neutral_count": 0,
            "total_mentions": 0, "sentiment_label": "无数据",
        }

    @staticmethod
    def _label(score: Decimal) -> str:
        if score > Decimal("0.3"):
            return "强烈看多"
        if score > Decimal("0.1"):
            return "偏积极"
        if score > Decimal("-0.1"):
            return "中性"
        if score > Decimal("-0.3"):
            return "偏消极"
        return "强烈看空"

    @staticmethod
    def _market_label(avg_score: Decimal) -> str:
        if avg_score > Decimal("0.2"):
            return "乐观"
        if avg_score > Decimal("0.05"):
            return "偏乐观"
        if avg_score > Decimal("-0.05"):
            return "中性"
        if avg_score > Decimal("-0.2"):
            return "偏悲观"
        return "悲观"
