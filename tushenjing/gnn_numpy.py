"""
纯 NumPy GNN 实现 — GCN + GAT（含真正注意力机制）。

当 PyTorch/PyG 不可用时的降级方案。
"""

import numpy as np


class NumPyGCN:
    """纯 NumPy GCN 实现。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(in_dim, hidden_dim).astype(np.float32) * 0.01
        self.W2 = rng.randn(hidden_dim, out_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray, adj_norm: np.ndarray) -> np.ndarray:
        h = adj_norm @ x @ self.W1 + self.b1
        h = np.maximum(0, h)
        h = adj_norm @ h @ self.W2 + self.b2
        return h

    def get_weights_snapshot(self) -> dict:
        return {"W1": self.W1.copy(), "W2": self.W2.copy()}

    def load_weights_snapshot(self, snapshot: dict) -> None:
        self.W1 = snapshot["W1"]
        self.W2 = snapshot["W2"]


class NumPyGAT:
    """纯 NumPy GAT — P1-3: 实现真正的单头注意力机制。

    e_ij = LeakyReLU(a_left^T @ Wh_i + a_right^T @ Wh_j)
    α_ij = softmax_j(e_ij)
    h'_i = Σ_j α_ij @ Wh_j
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        rng = np.random.RandomState(42)
        self.W = rng.randn(in_dim, hidden_dim).astype(np.float32) * 0.01
        self.a_left = rng.randn(hidden_dim, 1).astype(np.float32) * 0.01
        self.a_right = rng.randn(hidden_dim, 1).astype(np.float32) * 0.01
        self.W_out = rng.randn(hidden_dim, out_dim).astype(np.float32) * 0.01
        self.leaky_relu_slope = 0.2

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        h = x @ self.W  # [N, hidden]
        h_out = np.zeros((N, h.shape[1]), dtype=np.float32)

        for i in range(N):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) == 0:
                h_out[i] = h[i]
            else:
                scores = []
                h_i = h[i]
                for j in neighbors:
                    h_j = h[j]
                    e_ij = float((h_i @ self.a_left + h_j @ self.a_right).item())
                    e_ij = e_ij if e_ij > 0 else self.leaky_relu_slope * e_ij
                    scores.append(e_ij)

                scores = np.array(scores, dtype=np.float32)
                scores = np.exp(scores - scores.max())
                alpha = scores / scores.sum()

                for idx, j in enumerate(neighbors):
                    h_out[i] += alpha[idx] * h[j]

        return h_out @ self.W_out

    def get_weights_snapshot(self) -> dict:
        return {"W": self.W.copy(), "a_left": self.a_left.copy(), "a_right": self.a_right.copy(), "W_out": self.W_out.copy()}

    def load_weights_snapshot(self, snapshot: dict) -> None:
        self.W = snapshot["W"]
        self.a_left = snapshot["a_left"]
        self.a_right = snapshot["a_right"]
        self.W_out = snapshot["W_out"]
