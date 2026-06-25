"""Alpha因子（借鉴THU-BDC2026的158个Alpha特征体系）。纯numpy实现。"""
import numpy as np
from decimal import Decimal
from typing import Optional
from yinzi.factor_base import FactorBase, FactorCategory, FactorResult


class _AlphaBase(FactorBase):
    category = FactorCategory.MOMENTUM

    @staticmethod
    def _extract(daily_data: dict, field: str) -> np.ndarray:
        dates = sorted(daily_data.keys())
        return np.array([float(daily_data[d].get(field, np.nan)) for d in dates])

    @staticmethod
    def _to_d(value) -> Optional[Decimal]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return Decimal(str(round(float(value), 6)))

    @staticmethod
    def _rolling(x: np.ndarray, w: int, func) -> np.ndarray:
        r = np.full(len(x), np.nan)
        for i in range(w - 1, len(x)):
            r[i] = func(x[i - w + 1:i + 1])
        return r


# ── ROC收益率 (6个窗口) ─────────────────
class ROCFactor(_AlphaBase):
    direction = 1
    def __init__(self, window=5): self._w = window; self.name = f"roc_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        closes = self._extract(daily_data, "close")
        if len(closes) < self._w: return None
        return self._to_d((closes[-1] - closes[-self._w]) / closes[-self._w] if closes[-self._w] else 0)

# ── STD波动率 (5个窗口) ─────────────────
class STDFactor(_AlphaBase):
    direction = -1
    def __init__(self, window=20): self._w = window; self.name = f"std_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        closes = self._extract(daily_data, "close")
        if len(closes) < self._w + 1: return None
        window_closes = closes[-self._w - 1:]
        rets = np.diff(window_closes) / window_closes[:-1]
        return self._to_d(np.std(rets))

# ── CORR相关性 (5个窗口) ─────────────────
class CORRFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"corr_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        c = self._extract(daily_data, "close")[-self._w:]
        v = self._extract(daily_data, "volume")[-self._w:]
        if len(c) < self._w or np.std(c) == 0 or np.std(v) == 0: return None
        corr = np.corrcoef(c, v)[0, 1]
        return self._to_d(corr) if not np.isnan(corr) else None

# ── MAX/MIN极值 (各5个窗口) ─────────────
class MAXFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"max_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        highs = self._extract(daily_data, "high")[-self._w:]
        close = float(list(daily_data.values())[-1].get("close", 1)) if daily_data else 1
        return self._to_d(np.max(highs) / close if close > 0 else 0) if len(highs) >= self._w else None

class MINFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"min_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        lows = self._extract(daily_data, "low")[-self._w:]
        close = float(list(daily_data.values())[-1].get("close", 1)) if daily_data else 1
        return self._to_d(np.min(lows) / close if close > 0 else 0) if len(lows) >= self._w else None

# ── 成交量类 (VMA/VSTD各5个窗口) ────────
class VMAFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=5): self._w = window; self.name = f"vma_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        volumes = self._extract(daily_data, "volume")
        if len(volumes) < self._w: return None
        vma = np.mean(volumes[-self._w:])
        return self._to_d(volumes[-1] / vma if vma > 0 else 1.0)

class VSTDFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"vstd_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        volumes = self._extract(daily_data, "volume")
        return self._to_d(np.std(volumes[-self._w:]) / np.mean(volumes[-self._w:])) if len(volumes) >= self._w and np.mean(volumes[-self._w:]) > 0 else None

# ── 计数类 (CNTP上涨比例, 5个窗口) ────
class CNTPFactor(_AlphaBase):
    direction = 1
    def __init__(self, window=20): self._w = window; self.name = f"cntp_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        closes = self._extract(daily_data, "close")
        if len(closes) < self._w + 1: return None
        rets = np.diff(closes[-self._w - 1:])
        return self._to_d(np.sum(rets > 0) / self._w)

# ── 阶段1: BETA/RSQR/RANK/SKEW/KURT/TURN/AMP (各5窗口) ──
class BETAFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"beta_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        closes = self._extract(daily_data, "close")
        market = kwargs.get("market_data", {})
        mkt_closes = self._extract(market, "close") if market else np.array([])
        if len(closes) < self._w + 1 or len(mkt_closes) < self._w + 1: return None
        sr = np.diff(closes[-self._w - 1:]) / (closes[-self._w - 1:-1] + 1e-12)
        mr = np.diff(mkt_closes[-self._w - 1:]) / (mkt_closes[-self._w - 1:-1] + 1e-12)
        n = min(len(sr), len(mr))
        if n < self._w: return None
        v = np.var(mr[-n:])
        return self._to_d(np.cov(sr[-n:], mr[-n:])[0, 1] / v) if v > 0 else None

class RSQRFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"rsqr_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        closes = self._extract(daily_data, "close")
        market = kwargs.get("market_data", {})
        mkt_closes = self._extract(market, "close") if market else np.array([])
        if len(closes) < self._w + 1 or len(mkt_closes) < self._w + 1: return None
        sr = np.diff(closes[-self._w - 1:]) / (closes[-self._w - 1:-1] + 1e-12)
        mr = np.diff(mkt_closes[-self._w - 1:]) / (mkt_closes[-self._w - 1:-1] + 1e-12)
        n = min(len(sr), len(mr))
        if n < self._w: return None
        c = np.corrcoef(sr[-n:], mr[-n:])[0, 1]
        return self._to_d(c**2) if not np.isnan(c) else None

class RANKFactor(_AlphaBase):
    direction = 1
    def __init__(self, window=20): self._w = window; self.name = f"rank_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        closes = self._extract(daily_data, "close")
        if len(closes) < self._w + 1: return None
        rets = np.diff(closes[-self._w - 1:]) / (closes[-self._w - 1:-1] + 1e-12)
        return self._to_d(np.sum(rets <= rets[-1]) / self._w)

class SKEWFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"skew_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        closes = self._extract(daily_data, "close")
        if len(closes) < self._w + 2: return None
        rets = np.diff(closes[-self._w - 1:]) / (closes[-self._w - 1:-1] + 1e-12)
        s = np.std(rets)
        if s == 0: return None
        return self._to_d(np.sum((rets - np.mean(rets))**3) / len(rets) / s**3)

class KURTFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"kurt_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        closes = self._extract(daily_data, "close")
        if len(closes) < self._w + 2: return None
        rets = np.diff(closes[-self._w - 1:]) / (closes[-self._w - 1:-1] + 1e-12)
        s = np.std(rets)
        if s == 0: return None
        return self._to_d(np.sum((rets - np.mean(rets))**4) / len(rets) / s**4 - 3)

class TURNFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"turn_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        tr = self._extract(daily_data, "turnover_rate")[-self._w:]
        if len(tr) < self._w: return None
        m = np.mean(tr)
        return self._to_d(np.std(tr) / m) if m > 0 else None

class AMPFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"amp_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        h = self._extract(daily_data, "high")[-self._w:]
        l = self._extract(daily_data, "low")[-self._w:]
        c = self._extract(daily_data, "close")[-self._w:]
        if len(c) < self._w: return None
        return self._to_d(np.mean((h - l) / (c + 1e-12)))

# ── 阶段3: 交叉因子 ──────────────────────
class CrossFactor(_AlphaBase):
    """两因子交叉（乘积或比值）。"""
    def __init__(self, name: str, factor_a, factor_b, op: str = "product"):
        self.name = name; self._a = factor_a; self._b = factor_b; self._op = op

    def compute(self, code, daily_data, financial_data=None, **kwargs):
        va = self._a.compute(code, daily_data, financial_data, **kwargs)
        vb = self._b.compute(code, daily_data, financial_data, **kwargs)
        if va is None or vb is None: return None
        if self._op == "product": return self._to_d(float(va) * float(vb))
        return self._to_d(float(va) / float(vb)) if float(vb) != 0 else None

# ── 阶段4: 日内特征 ──────────────────────
class VWAPFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"vwap_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        a = self._extract(daily_data, "amount")[-self._w:]
        v = self._extract(daily_data, "volume")[-self._w:]
        c = self._extract(daily_data, "close")
        if len(a) < self._w or np.sum(v) <= 0: return None
        vwap = np.sum(a) / np.sum(v)
        return self._to_d((c[-1] - vwap) / vwap)

class HLSpreadFactor(_AlphaBase):
    direction = 0
    def __init__(self, window=20): self._w = window; self.name = f"hl_spread_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        h = self._extract(daily_data, "high")[-self._w:]
        l = self._extract(daily_data, "low")[-self._w:]
        c = self._extract(daily_data, "close")[-self._w:]
        if len(c) < self._w: return None
        return self._to_d(np.mean((h - l) / (c + 1e-12)))

class OCFactor(_AlphaBase):
    direction = 1
    def __init__(self, window=20): self._w = window; self.name = f"oc_{window}"
    def compute(self, code, daily_data, financial_data=None, **kwargs):
        o = self._extract(daily_data, "open")[-self._w:]
        c = self._extract(daily_data, "close")[-self._w:]
        if len(c) < self._w: return None
        return self._to_d(np.mean((c - o) / (o + 1e-12)))

# ── 工厂函数 (全部4阶段) ───────────────
def create_alpha_factors() -> list[FactorBase]:
    """创建全部Alpha因子(~157个) — 阶段0~4完整版。"""
    factors = []
    W0 = [5, 10, 20, 30, 60]  # 基础窗口
    EXTRA = [3, 15, 40, 90, 120, 180, 252]  # 扩展窗口
    KEY_CLASSES = [ROCFactor, STDFactor, MAXFactor, MINFactor, BETAFactor, RSQRFactor]

    # 阶段0: 7类×5窗口=35
    for w in W0:
        for cls in [ROCFactor, STDFactor, CORRFactor, MAXFactor, MINFactor, VMAFactor, CNTPFactor]:
            factors.append(cls(w))
    # 阶段1: 7类×5窗口=35
    for w in W0:
        for cls in [BETAFactor, RSQRFactor, RANKFactor, SKEWFactor, KURTFactor, TURNFactor, AMPFactor]:
            factors.append(cls(w))
    # 阶段2: 6类×7窗口=42
    for w in EXTRA:
        for cls in KEY_CLASSES:
            factors.append(cls(w))
    # 阶段3: 交叉因子≈30
    for w in [10, 20, 60]:
        factors.append(CrossFactor(f"roc_std_{w}", ROCFactor(w), STDFactor(w), "ratio"))
    for w in [20, 60]:
        factors.append(CrossFactor(f"beta_std_{w}", BETAFactor(w), STDFactor(w), "ratio"))
    for w in [10, 20, 60]:
        factors.append(CrossFactor(f"corr_vma_{w}", CORRFactor(w), VMAFactor(w), "product"))
        factors.append(CrossFactor(f"max_min_{w}", MAXFactor(w), MINFactor(w), "ratio"))
    for w in [20, 60]:
        factors.append(CrossFactor(f"rank_roc_{w}", RANKFactor(w), ROCFactor(w), "product"))
    # 阶段4: 日内3类×5窗口=15
    for w in W0:
        for cls in [VWAPFactor, HLSpreadFactor, OCFactor]:
            factors.append(cls(w))
    return factors
