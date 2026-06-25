"""标签构建工具 — P2-6: 前视开盘收益率。label=(open[t+h]-open[t+1])/open[t+1]"""
from decimal import Decimal


def build_forward_returns(daily_data_map: dict[str, dict], horizon: int = 5) -> dict[str, Decimal]:
    """构建前视收益率标签（使用开盘价，避免收盘价可交易性偏差）。"""
    results = {}
    for code, bars in daily_data_map.items():
        dates = sorted(bars.keys())
        if len(dates) < horizon + 2: continue
        open_t1 = float(bars[dates[-(horizon + 1)]].get("open", 0))
        open_th = float(bars[dates[-1]].get("open", 0))
        if open_t1 > 1e-4: results[code] = Decimal(str(round((open_th - open_t1) / open_t1, 6)))
    return results
