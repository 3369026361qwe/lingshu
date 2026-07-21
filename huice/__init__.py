"""huice — 回测系统 (v4.2). 事件驱动回测+VaR检验+数据窥探防御+绩效报告+网格搜索+归因分析+交叉验证。"""
from huice.attribution import AttributionEngine, BrinsonAttribution, FactorAttribution, RiskAttribution
from huice.backtest_engine import BacktestEngine, DBBacktestRunner, PerformanceReport
from huice.cross_validation import CVResult, PurgedKFold, WalkForwardCV, compute_cv_summary
from huice.data_snooping import DataSnoopingDefender
from huice.grid_search import GridSearch
from huice.performance_metrics import PerformanceMetrics
from huice.report_generator import ReportGenerator

__all__ = [
    "BacktestEngine", "DBBacktestRunner", "PerformanceReport",
    "PerformanceMetrics", "ReportGenerator", "GridSearch",
    "AttributionEngine", "BrinsonAttribution", "FactorAttribution", "RiskAttribution",
    "DataSnoopingDefender",
    "PurgedKFold", "WalkForwardCV", "CVResult", "compute_cv_summary",
]
