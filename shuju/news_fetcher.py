"""
新闻/公告数据获取器。

功能:
    - 上市公司公告抓取
    - 财经新闻采集
    - 关键词过滤与分类

特性:
    - 默认使用 AKShare 免费新闻接口
    - 支持扩展付费新闻 API
    - 优雅降级 (无网络时返回空列表)

Usage:
    fetcher = NewsFetcher()
    news = fetcher.get_stock_news("000001", limit=20)
"""

import hashlib
import logging
import time

from shuju.cache_manager import DataCacheManager

_logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_REQUEST_INTERVAL = 0.5


class NewsFetcher:
    """新闻公告数据获取器。"""

    def __init__(self, cache: DataCacheManager | None = None) -> None:
        self._cache = cache or DataCacheManager()
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < _REQUEST_INTERVAL:
            time.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request = time.time()

    # ── 个股新闻 ────────────────────────────────────────

    def get_stock_news(self, code: str, limit: int = 20) -> list[dict]:
        """获取个股相关新闻。

        Returns:
            [{"id": "md5hash", "title": "...", "content": "...",
              "source": "东方财富", "pub_time": "2026-06-01 15:30",
              "url": "https://..."}, ...]
        """
        try:
            import akshare as ak
            self._rate_limit()

            # AKShare 个股新闻接口
            df = ak.stock_news_em(symbol=code)
            if df is None or df.empty:
                return []

            results = []
            for _, row in df.head(limit).iterrows():
                try:
                    title = str(row.get("title", "") or row.get("标题", ""))
                    content = str(row.get("content", "") or row.get("内容", ""))
                    if not title:
                        continue

                    news_id = hashlib.md5(
                        f"{code}:{title}".encode()
                    ).hexdigest()[:16]

                    item = {
                        "id": news_id,
                        "code": code,
                        "title": title,
                        "content": content[:500] if content else "",
                        "source": str(row.get("source", "") or row.get("来源", "")),
                        "pub_time": str(row.get("pub_time", "") or row.get("发布时间", "")),
                        "url": str(row.get("url", "") or row.get("链接", "")),
                    }
                    results.append(item)

                    # 缓存每条新闻
                    self._cache.cache_news(news_id, item)
                except Exception as exc:
                    _logger.debug("Skip news row: %s", exc)

            return results
        except ImportError:
            _logger.warning("akshare not available for news")
        except Exception as exc:
            _logger.warning("Failed to fetch news for %s: %s", code, exc)

        return []

    # ── 公告 ────────────────────────────────────────────

    def get_announcements(self, code: str, limit: int = 10) -> list[dict]:
        """获取上市公司公告。

        Returns:
            [{"id": "...", "title": "...", "type": "年报/季报/临时公告",
              "pub_date": "2026-06-01", "url": "..."}, ...]
        """
        try:
            import akshare as ak
            self._rate_limit()

            df = ak.stock_notice_report(symbol=code)
            if df is None or df.empty:
                return []

            results = []
            for _, row in df.head(limit).iterrows():
                try:
                    title = str(row.get("title", "") or row.get("公告标题", ""))
                    if not title:
                        continue

                    item = {
                        "id": hashlib.md5(f"{code}:{title}".encode()).hexdigest()[:16],
                        "code": code,
                        "title": title,
                        "pub_date": str(row.get("pub_date", "") or row.get("公告日期", "")),
                        "url": str(row.get("url", "") or row.get("链接", "")),
                        "type": self._classify_announcement(title),
                    }
                    results.append(item)
                except Exception as exc:
                    _logger.debug("Skip announcement: %s", exc)

            return results
        except ImportError:
            _logger.warning("akshare not available for announcements")
        except Exception as exc:
            _logger.warning("Failed to fetch announcements for %s: %s", code, exc)

        return []

    # ── 全市场新闻摘要 ──────────────────────────────────

    def get_market_news_summary(self, limit: int = 50) -> list[dict]:
        """获取全市场重要新闻摘要。"""
        try:
            import akshare as ak
            self._rate_limit()

            # 财联社电报 (全市场快讯)
            df = ak.stock_info_global_cls()
            if df is None or df.empty:
                return []

            results = []
            for _, row in df.head(limit).iterrows():
                try:
                    content = str(row.get("content", "") or row.get("内容", ""))
                    if not content:
                        continue
                    results.append({
                        "id": hashlib.md5(content.encode()).hexdigest()[:16],
                        "content": content[:300],
                        "time": str(row.get("time", "") or row.get("时间", "")),
                    })
                except Exception:
                    continue
            return results
        except ImportError:
            _logger.warning("akshare not available for market news")
        except Exception as exc:
            _logger.warning("Failed to fetch market news: %s", exc)

        return []

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _classify_announcement(title: str) -> str:
        """根据公告标题分类。"""
        if any(kw in title for kw in ["年报", "年度报告", "annual report"]):
            return "年报"
        if any(kw in title for kw in ["季报", "季度报告", "quarterly"]):
            return "季报"
        if any(kw in title for kw in ["业绩预告", "业绩快报"]):
            return "业绩预告"
        if any(kw in title for kw in ["分红", "派息", "送转"]):
            return "分红公告"
        if any(kw in title for kw in ["增持", "减持", "回购"]):
            return "股东变动"
        if any(kw in title for kw in ["重组", "并购", "资产"]):
            return "重大事项"
        if any(kw in title for kw in ["诉讼", "处罚", "监管"]):
            return "风险提示"
        return "其他"
