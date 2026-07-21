"""参数网格搜索 — 遍历参数组合，记录全部实验结果。支持 Walk-Forward CV。"""
import itertools
import logging

from huice.backtest_engine import BacktestEngine
from huice.cross_validation import CVResult, PurgedKFold, WalkForwardCV, compute_cv_summary

_logger = logging.getLogger(__name__)


class GridSearch:
    """参数网格搜索。每个参数组合产生独立实验记录。"""

    def __init__(self, backtest_engine: BacktestEngine):
        self._engine = backtest_engine

    def search(
        self, base_config: dict, param_grid: dict,
        cv: str | None = None,
        cv_config: dict | None = None,
    ) -> list[dict]:
        """遍历参数网格，返回全部实验结果。

        Args:
            base_config: 基础配置 (start_date, end_date, initial_capital, ...)
            param_grid: {param_name: [value1, value2, ...]}
                        例如: {"top_n": [10, 20, 30], "risk_aversion": [2.0, 3.0, 5.0]}
            cv: 交叉验证模式 — None (单次 split), "purged_kfold", "walkforward"
            cv_config: CV 参数 — e.g. {"n_splits": 5, "embargo_pct": 0.01}

        Returns:
            [{experiment_id, params, metrics, cv_summary?}, ...]
            每次实验的完整结果。当 cv 启用时，metrics 为全量回测结果，
            额外包含 cv_summary 字段。
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo, strict=False))
            config = {**base_config}
            for k, v in params.items():
                if hasattr(config.get("signal_generator"), k):
                    setattr(config["signal_generator"], k, v)
                config[k] = v

            report = self._engine.run(config)
            report["grid_params"] = params

            # 如果启用了 CV，做交叉验证回测
            if cv:
                cv_results = self._run_cross_validation(config, cv, cv_config or {})
                report["cv_summary"] = compute_cv_summary(cv_results)
                report["cv_details"] = [
                    {
                        "fold": r.fold,
                        "train_start": r.train_start,
                        "train_end": r.train_end,
                        "test_start": r.test_start,
                        "test_end": r.test_end,
                        "sharpe": r.sharpe,
                        "total_return": r.total_return,
                        "max_drawdown": r.max_drawdown,
                    }
                    for r in cv_results
                ]

            results.append(report)

        # 按夏普（或 CV 夏普均值）排序
        if cv:
            results.sort(
                key=lambda r: (
                    r.get("cv_summary", {}).get("sharpe_mean")
                    if r.get("cv_summary", {}).get("sharpe_mean") is not None
                    else -999
                ),
                reverse=True,
            )
        else:
            results.sort(
                key=lambda r: r.get("metrics", {}).get("sharpe") or -999,
                reverse=True,
            )
        return results

    def _run_cross_validation(
        self, config: dict, cv_mode: str, cv_config: dict,
    ) -> list[CVResult]:
        """对单个参数组合运行交叉验证。

        Args:
            config: 回测配置
            cv_mode: "purged_kfold" | "walkforward"
            cv_config: CV 超参数

        Returns:
            CVResult 列表，每个 fold 一个
        """
        # 获取交易日列表
        dates = config["data_loader"].get_trade_dates(
            config.get("start_date", ""), config.get("end_date", ""),
        )
        if len(dates) < 60:
            _logger.warning("Too few dates (%d) for CV, skipping", len(dates))
            return []

        # 构建 CV splitter
        if cv_mode == "purged_kfold":
            splitter = PurgedKFold(
                n_splits=cv_config.get("n_splits", 5),
                embargo_pct=cv_config.get("embargo_pct", 0.01),
                purge_pct=cv_config.get("purge_pct", 0.01),
            )
        elif cv_mode == "walkforward":
            splitter = WalkForwardCV(
                min_train_size=cv_config.get("min_train_size", 252),
                step_size=cv_config.get("step_size", 63),
                test_size=cv_config.get("test_size", 63),
                expanding=cv_config.get("expanding", True),
            )
        else:
            raise ValueError(f"Unknown cv mode: {cv_mode}")

        cv_results = []
        try:
            folds = list(splitter.split(dates))
        except ValueError as exc:
            _logger.warning("CV split failed: %s", exc)
            return []

        for fold_idx, (train_dates, test_dates) in enumerate(folds):
            if len(train_dates) < 10 or len(test_dates) < 5:
                continue

            train_start = train_dates[0]
            train_end = train_dates[-1]
            test_start = test_dates[0]
            test_end = test_dates[-1]

            # 测试集回测（验证参数在未见数据上的表现）
            test_config = {**config}
            test_config["start_date"] = test_start
            test_config["end_date"] = test_end
            try:
                test_report = self._engine.run(test_config)
            except Exception as exc:
                _logger.warning("Fold %d test failed: %s", fold_idx, exc)
                continue

            test_metrics = test_report.get("metrics", {})
            cv_results.append(CVResult(
                fold=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                sharpe=test_metrics.get("sharpe"),
                total_return=test_metrics.get("total_return"),
                max_drawdown=test_metrics.get("max_drawdown"),
            ))

        return cv_results

    @staticmethod
    def best_params(results: list[dict]) -> dict:
        """从网格搜索结果中提取最佳参数组合。"""
        if not results:
            return {}
        best = results[0]
        metrics = best.get("metrics", {})
        return {
            "experiment_id": best["experiment_id"],
            "params": best.get("grid_params", {}),
            "sharpe": metrics.get("sharpe"),
            "total_return": metrics.get("total_return"),
        }
