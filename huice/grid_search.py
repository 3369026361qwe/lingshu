"""参数网格搜索 — 遍历参数组合，记录全部实验结果。"""
import itertools
from huice.backtest_engine import BacktestEngine


class GridSearch:
    """参数网格搜索。每个参数组合产生独立实验记录。"""

    def __init__(self, backtest_engine: BacktestEngine):
        self._engine = backtest_engine

    def search(self, base_config: dict, param_grid: dict) -> list[dict]:
        """遍历参数网格，返回全部实验结果。

        Args:
            base_config: 基础配置 (start_date, end_date, initial_capital, ...)
            param_grid: {param_name: [value1, value2, ...]}
                        例如: {"top_n": [10, 20, 30], "risk_aversion": [2.0, 3.0, 5.0]}

        Returns:
            [{experiment_id, params, metrics}, ...]  每次实验的完整结果
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            config = {**base_config}
            for k, v in params.items():
                if hasattr(config.get("signal_generator"), k):
                    setattr(config["signal_generator"], k, v)
                config[k] = v

            report = self._engine.run(config)
            report["grid_params"] = params
            results.append(report)

        # 按夏普排序
        results.sort(key=lambda r: r.get("metrics", {}).get("sharpe") or -999, reverse=True)
        return results

    @staticmethod
    def best_params(results: list[dict]) -> dict:
        """从网格搜索结果中提取最佳参数组合。"""
        if not results: return {}
        best = results[0]
        return {"experiment_id": best["experiment_id"], "params": best.get("grid_params", {}), "sharpe": best.get("metrics", {}).get("sharpe"), "total_return": best.get("metrics", {}).get("total_return")}
