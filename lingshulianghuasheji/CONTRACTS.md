# CONTRACTS — 灵枢 v4.0 统一接口契约

本文件定义五角色协同工作的接口边界。所有模块的公共 API 必须严格遵循此契约。

---

## 数据类型

跨模块传递使用 `Decimal`（金融精度），`float` 仅限 GNN 训练。

### StockSignal

```python
@dataclass
class StockSignal:
    code: str
    date: str
    factor_score: Optional[Decimal] = None     # yinzi 路径
    gnn_score: Optional[Decimal] = None        # tushenjing 路径
    agent_score: Optional[Decimal] = None      # zhinengti 路径
    agent_confidence: Optional[Decimal] = None
    industry: Optional[str] = None
    market_cap: Optional[Decimal] = None
```

### RiskInput

```python
@dataclass
class RiskInput:
    returns: list[list[Decimal]]     # 资产回报矩阵 [资产][时间]
    weights: list[Decimal]           # 组合权重
    asset_codes: list[str]
    confidence_levels: list[Decimal] # [0.95, 0.99, 0.999]
    lookback_days: int = 252
```

### OptimizationInput

```python
@dataclass
class OptimizationInput:
    signals: list[StockSignal]
    risk_input: RiskInput
    current_weights: list[Decimal]
    constraints: dict
    views: Optional[list[View]] = None  # BL views
```

### PerformanceReport

```python
@dataclass
class PerformanceReport:
    total_return: Decimal
    annualized_return: Decimal
    annualized_volatility: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    var_95: Decimal
    var_99: Decimal
    cvar_95: Decimal
    cvar_99: Decimal
    var_backtest: dict           # Kupiec/Christoffersen results
    deflated_sharpe: float       # DSR p-value
    probabilistic_sharpe: float  # PSR
    attribution: dict            # Brinson + Factor + Risk
```

---

## jingsuan/<br/>合约 (精算师)

### EVTEngine

| 方法 | 输入 | 输出 | 合约 |
|------|------|------|------|
| `fit_gpd(returns, q, method)` | `list[Decimal]`, `Decimal`, `str` | `EVTFitResult` | `q` 默认 0.90 |
| `tail_var(fit, n, levels)` | `EVTFitResult`, `int`, `list[Decimal]` | `EVTVaRResult` | levels 默认 [0.95,0.99,0.999] |
| `hill_estimator(returns, frac)` | `list[Decimal]`, `Decimal` | `Decimal` | frac 默认 0.10 |
| `compare_models(returns)` | `list[Decimal]` | `dict` | Normal vs EVT vs Historical |

### CopulaEngine

| 方法 | 输入 | 输出 | 合约 |
|------|------|------|------|
| `fit(returns_matrix, types)` | `list[list[Decimal]]`, `list[CopulaType]` | `CopulaFit` | 返回 AIC 最优 |
| `simulate(fit, n, m)` | `CopulaFit`, `int`, `int` | `list[list[Decimal]]` | n 默认 10000 |
| `portfolio_tail_loss(fit, w, scenarios, conf)` | 4个参数 | `Decimal` | conf 默认 0.99 |

### RuinEngine

| 方法 | 输入 | 输出 | 合约 |
|------|------|------|------|
| `estimate_ruin_prob(returns, size, config)` | 3个参数 | `Decimal` | 蒙特卡洛 100k sims |
| `optimal_position_size(returns, config)` | 2个参数 | `Decimal` | 二分搜索 |
| `dynamic_risk_budget(drawdown, config)` | `Decimal`, `RuinConfig` | `Decimal` | 回撤越大→仓位越小 |

### CredibilityEngine

| 方法 | 输入 | 输出 | 合约 |
|------|------|------|------|
| `buhlmann_straub(sources)` | `list[SourceTrackRecord]` | `CredibilityWeights` | Bühlmann-Straub 信度 |
| `fuse_signals(sources, signals)` | 2个参数 | `list[Decimal]` | 信度加权融合得分 |

---

## juece/<br/>合约 (量化金工)

### PortfolioOptimizer

| 方法 | 输入 | 输出 | 合约 |
|------|------|------|------|
| `estimate_covariance(returns)` | `list[list[Decimal]]` | `list[list[Decimal]]` | Ledoit-Wolf 收缩 |
| `implied_returns(cov, mkt_w)` | 2个参数 | `list[Decimal]` | Π = δ·Σ·w_mkt |
| `incorporate_views(pi, cov, views)` | 3个参数 | `tuple[list, list[list]]` | BL 贝叶斯更新 |
| `optimize(post_r, post_cov, constraints)` | 3个参数 | `BLOptimizationResult` | cvxpy/scipy |

### HRPOptimizer

| 方法 | 输入 | 输出 | 合约 |
|------|------|------|------|
| `cluster(returns)` | `list[list[Decimal]]` | `dict` | 层次聚类 |
| `allocate(returns, tree)` | 2个参数 | `HRPResult` | 递归二等分 |

### MarketImpactModel

| 方法 | 输入 | 输出 | 合约 |
|------|------|------|------|
| `estimate(v, V, sigma)` | 3个参数 | `Decimal` | Almgren-Chriss 公式 |

---

## yinzi/ + huice/ 合约 (计量经济学家)

### MultipleTesting

`bonferroni(pvalues, a=0.05)` `→` `MultipleTestingResult`
`holm_bonferroni(pvalues, a=0.05)` `→` `MultipleTestingResult`
`benjamini_hochberg(pvalues, a=0.05)` `→` `MultipleTestingResult`

### DataSnooping

`deflated_sharpe(sr, n_trials, T)` `→` `float` (p-value)
`haircut_sharpe(sr, n_trials, T)` `→` `float`
`probabilistic_sharpe(sr, sr_benchmark, T, skew, kurt)` `→` `float`

### GARCH

`garch_11(returns)` `→` `GARCHResult` (ω, α, β, conditional_vol)
`egarch_11(returns)` `→` `GARCHResult` (with γ leverage effect)
`gjr_garch(returns)` `→` `GARCHResult` (with asymmetric term)

### Regime

`hmm_fit(returns, n_states, features)` `→` `dict` (states, transitions, means)
`hmm_predict(raw, model)` `→` `int`
`regime_conditional_var(returns, labels, conf)` `→` `dict`

---

## 横切层合约

### ConfigLoader (`shujuku/settings.py`)

`load_config(path="config.yaml")` `→` `AppConfig` — 单例缓存

### SafeMath (`shuju/utils.py`)

`safe_divide(num, den, default=0)` `→` `Decimal`
`safe_mean(values, default=0)` `→` `Decimal`
`safe_pct_change(old, new, default=0)` `→` `Decimal`

### Metrics (`shujuku/metrics.py`)

`registry()` `→` `CollectorRegistry` — 统一 Prometheus 注册中心

---

## 调用链示意

```
juece/PortfolioOptimizer.optimize()
  ├── jingsuan/EVTEngine.tail_var()         ← 尾部风险约束
  ├── jingsuan/RuinEngine.dynamic_budget()  ← 动态仓位上限
  └── jingsuan/CredibilityEngine.fuse()     ← 信号融合权重

juece/PortfolioOptimizer.estimate_covariance()
  └── 内部 Ledoit-Wolf (不外调)

huice/BacktestEngine.run()
  ├── yinzi/MultipleTesting.bonferroni()    ← 因子筛选
  ├── huice/DataSnooping.deflated_sharpe()  ← 数据窥探防御
  └── jingsuan/VaRBacktestSuite.run_all()   ← VaR 回测

所有模块:
  └── shuju/utils.safe_divide()             ← 全局安全除法
  └── shujuku/settings.load_config()        ← 全局配置
```

---

## 测试合约

每个遵守本契约的模块必须包含：

1. **合成数据验证** — 已知分布 → 检查参数恢复
2. **边界情况** — 空列表、零值、单元素
3. **动态范围检查** — 输出在合理区间内
