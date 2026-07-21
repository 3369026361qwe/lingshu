"""
Walk-Forward 交叉验证 — 时间序列专用 CV.

实现:
    PurgedKFold    — Purged K-Fold: 连续分割 + embargo 间隙，防止信息泄露
    WalkForwardCV  — 滚动训练/测试窗口，模拟真实部署场景

参考:
    Bailey & López de Prado (2017). The Deflated Sharpe Ratio.
    Advances in Financial Machine Learning, Chapter 7.

Usage:
    from huice.cross_validation import PurgedKFold, WalkForwardCV

    # Purged K-Fold
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    for train_idx, test_idx in pkf.split(dates):
        train_dates = [dates[i] for i in train_idx]
        test_dates = [dates[i] for i in test_idx]

    # Walk-Forward
    wf = WalkForwardCV(min_train_size=252, step_size=63, test_size=63)
    for train_dates, test_dates in wf.split(dates):
        ...
"""

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class CVResult:
    """单次交叉验证 fold 的结果。"""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    sharpe: float | None
    total_return: float | None
    max_drawdown: float | None


class PurgedKFold:
    """Purged K-Fold 时间序列交叉验证.

    标准 K-Fold 在时间序列上有信息泄露（训练集包含未来信息）。
    Purged K-Fold 通过以下方式解决:
        1. 连续分割（不打乱）
        2. Purge: 训练集和测试集之间移除重叠标签
        3. Embargo: 训练集结束后添加禁运期

    数学:
        n = len(dates)
        fold_size = n // n_splits
        embargo_size = int(embargo_pct * n)
        purge_size = int(purge_pct * n)
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 0 <= embargo_pct < 0.5:
            raise ValueError(f"embargo_pct must be in [0, 0.5), got {embargo_pct}")
        if not 0 <= purge_pct < 0.5:
            raise ValueError(f"purge_pct must be in [0, 0.5), got {purge_pct}")

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(self, dates: list[str]) -> Iterator[tuple[list[str], list[str]]]:
        """生成 (train_dates, test_dates) 迭代器.

        Args:
            dates: 已排序的交易日列表

        Yields:
            (train_dates, test_dates) — 每个 fold 一份
        """
        n = len(dates)
        if n < self.n_splits * 3:
            raise ValueError(
                f"Need at least {self.n_splits * 3} dates for {self.n_splits}-fold CV, "
                f"got {n}"
            )

        embargo_size = max(1, int(self.embargo_pct * n))
        purge_size = max(1, int(self.purge_pct * n))
        fold_size = n // self.n_splits

        for fold in range(self.n_splits):
            # 测试集: 第 fold 个连续区间
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, n) if fold < self.n_splits - 1 else n

            # 训练集: 测试集开始之前的所有历史数据
            # embargo 间隙: 训练集末尾与测试集之间保留 embargo_size 天
            # purge: 从训练集末尾再删 purge_size 天（防标签重叠）
            train_cutoff = test_start - embargo_size
            if train_cutoff <= 0:
                continue  # fold 0 无历史训练数据，跳过

            # 训练集末尾应用 purge（靠近测试边界处删除标签重叠样本）
            train_end = min(train_cutoff - purge_size, train_cutoff)
            if train_end < 1:
                continue

            train_start = 0
            yield (
                dates[train_start:train_end],
                dates[test_start:test_end],
            )

    def split_indices(self, n: int) -> Iterator[tuple[list[int], list[int]]]:
        """生成 (train_indices, test_indices) 迭代器（按索引）。

        Args:
            n: 总观测数

        Yields:
            (train_indices, test_indices)
        """
        dates = [str(i) for i in range(n)]
        for train_d, test_d in self.split(dates):
            yield (
                [int(d) for d in train_d],
                [int(d) for d in test_d],
            )


class WalkForwardCV:
    """Walk-Forward (Rolling Window) 交叉验证.

    模拟真实策略部署:
        - 训练集: 不断扩大的历史窗口（或固定长度滚动窗口）
        - 测试集: 紧随其后的固定窗口
        - 逐步前进，每次 step_size 个周期

    数学:
        n = len(dates)
        fold 0: train=[0:min_train],            test=[min_train:min_train+test_size]
        fold 1: train=[0:min_train+step],        test=[min_train+step:min_train+step+test_size]
        fold k: train=[0:min_train+k*step],      test=[min_train+k*step:min_train+k*step+test_size]
    """

    def __init__(
        self,
        min_train_size: int = 252,
        step_size: int = 63,
        test_size: int = 63,
        expanding: bool = True,
    ):
        """
        Args:
            min_train_size: 最小训练窗口（交易日数）
            step_size: 每次前进步长
            test_size: 测试窗口大小
            expanding: True = 扩展窗口, False = 固定滚动窗口
        """
        if min_train_size < 10:
            raise ValueError(f"min_train_size must be >= 10, got {min_train_size}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")

        self.min_train_size = min_train_size
        self.step_size = step_size
        self.test_size = test_size
        self.expanding = expanding

    @property
    def n_splits(self) -> int | None:
        """运行时根据数据长度确定的 fold 数，初始化时为 None。"""
        return None  # 动态确定

    def split(self, dates: list[str]) -> Iterator[tuple[list[str], list[str]]]:
        """生成 (train_dates, test_dates) 迭代器.

        Args:
            dates: 已排序的交易日列表

        Yields:
            (train_dates, test_dates)
        """
        n = len(dates)
        if n < self.min_train_size + self.test_size:
            raise ValueError(
                f"Need at least {self.min_train_size + self.test_size} dates, got {n}"
            )

        train_end = self.min_train_size
        while train_end + self.test_size <= n:
            if self.expanding:
                train_dates = dates[:train_end]
            else:
                # 固定窗口: 最后 min_train_size 个观测
                train_start = max(0, train_end - self.min_train_size)
                train_dates = dates[train_start:train_end]

            test_dates = dates[train_end:train_end + self.test_size]
            yield (train_dates, test_dates)

            train_end += self.step_size


def compute_cv_summary(cv_results: list[CVResult]) -> dict:
    """从多个 fold 的 CVResult 计算聚合统计。

    Args:
        cv_results: CVResult 列表

    Returns:
        {
            n_folds, sharpe_mean, sharpe_std,
            total_return_mean, total_return_std,
            max_drawdown_mean, max_drawdown_std,
        }
    """
    if not cv_results:
        return {
            "n_folds": 0,
            "sharpe_mean": None, "sharpe_std": None,
            "total_return_mean": None, "total_return_std": None,
            "max_drawdown_mean": None, "max_drawdown_std": None,
        }

    sharpes = [r.sharpe for r in cv_results if r.sharpe is not None]
    returns = [r.total_return for r in cv_results if r.total_return is not None]
    drawdowns = [r.max_drawdown for r in cv_results if r.max_drawdown is not None]

    def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
        if not vals:
            return None, None
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = var ** 0.5
        else:
            std = 0.0
        return mean, std

    s_mean, s_std = _mean_std(sharpes)
    r_mean, r_std = _mean_std(returns)
    d_mean, d_std = _mean_std(drawdowns)

    return {
        "n_folds": len(cv_results),
        "sharpe_mean": round(s_mean, 4) if s_mean is not None else None,
        "sharpe_std": round(s_std, 4) if s_std is not None else None,
        "total_return_mean": round(r_mean, 4) if r_mean is not None else None,
        "total_return_std": round(r_std, 4) if r_std is not None else None,
        "max_drawdown_mean": round(d_mean, 4) if d_mean is not None else None,
        "max_drawdown_std": round(d_std, 4) if d_std is not None else None,
    }
