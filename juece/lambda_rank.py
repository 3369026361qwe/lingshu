"""LambdaRank排序损失 — P0-2: 借鉴THU-BDC2026。"""
import numpy as np


class LambdaRankLoss:
    """LambdaRank排序损失。对每对(i,j)，如果 label_i > label_j 但 pred_i < pred_j，施加ΔNDCG加权的梯度。"""

    def __init__(self, sigma: float = 1.0, k: int = 5):
        self.sigma = sigma
        self.k = k

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """1 - NDCG@k"""
        order = np.argsort(-y_pred)
        dcg = self._dcg(y_true[order], self.k)
        ideal = self._dcg(np.sort(y_true)[::-1], self.k)
        return 0.0 if ideal == 0 else 1.0 - dcg / ideal

    def compute_gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """LambdaRank梯度"""
        n = len(y_pred)
        grad = np.zeros(n)
        ideal = self._dcg(np.sort(y_true)[::-1], self.k)
        if ideal == 0:
            return grad
        for i in range(n):
            for j in range(n):
                if y_true[i] <= y_true[j]:
                    continue
                delta = self._delta_ndcg(y_true, i, j, ideal)
                if delta == 0:
                    continue
                sij = y_pred[i] - y_pred[j]
                rho = 1.0 / (1.0 + np.exp(-self.sigma * sij))
                lam = -self.sigma * delta * rho * (1 - rho)
                grad[i] += lam
                grad[j] -= lam
        return grad

    def _dcg(self, labels: np.ndarray, k: int) -> float:
        k = min(k, len(labels))
        gains = 2.0 ** labels[:k] - 1.0
        return float(np.sum(gains / np.log2(np.arange(2, k + 2))))

    def _delta_ndcg(self, labels: np.ndarray, i: int, j: int, ideal: float) -> float:
        return abs((2.0 ** labels[i] - 1.0) - (2.0 ** labels[j] - 1.0)) / ideal if ideal > 0 else 0.0
