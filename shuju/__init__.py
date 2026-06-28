"""
shuju — 数据层

多源数据采集 + 预处理管道 + 时间对齐 + 缓存管理。

数据源:
    AKShare  — 全市场日线行情、行业分类
    Tushare  — 财务报表、估值数据
    News API — 新闻公告
    Weibo    — 社交媒体舆情

预处理管道:
    清洗 → 对齐 → 去极值(MAD/σ) → 标准化(z-score) → 行业中性化

Usage:
    from shuju.akshare_fetcher import AKShareFetcher
    from shuju.data_preprocessor import DataPreprocessor
"""

from shuju.akshare_fetcher import AKShareFetcher
from shuju.cache_manager import DataCacheManager
from shuju.data_aligner import DataAligner
from shuju.data_preprocessor import DataPreprocessor
from shuju.news_fetcher import NewsFetcher
from shuju.sentiment_fetcher import SentimentFetcher
from shuju.tushare_fetcher import TushareFetcher

__all__ = [
    "AKShareFetcher",
    "TushareFetcher",
    "NewsFetcher",
    "SentimentFetcher",
    "DataPreprocessor",
    "DataAligner",
    "DataCacheManager",
]
