# CLAUDE.md — 量化金融工程师窗口 (灵枢 v4.0 feat/jin-gong-enhance)

## 角色: 量化金融工程师

你负责完善 `juece/` `huice/` `zhixing/` 的投资组合优化、回测引擎和交易执行。

## 约束
- **修改/新增 `juece/` `huice/` `zhixing/` 目录内的文件**
- 可以 import jingsuan 的接口（`from jingsuan.evt_engine import EVTEngine`）但**不修改** jingsuan 代码
- 所有金融计算用 Decimal，ML超参可用 float
- 每个新模块必须有独立验证

## 任务清单

### 1. 回测引擎重构 (huice/backtest_engine.py)
- [ ] 真正事件驱动架构 — 集成 MockBroker + OrderManager
- [ ] 使用 `from jingsuan.var_backtest import VaRBacktestSuite` 做 VaR 回测
- [ ] 使用 `from huice.data_snooping import DataSnoopingDefender` 输出 DSR/PSR
- [ ] 使用 `from huice.attribution import AttributionEngine` 输出归因分析

### 2. 组合优化深化 (juece/portfolio_optimizer.py)
- [ ] 用 cvxpy 实现完整的凸优化（替代当前解析解+投影）
- [ ] 集成 `from jingsuan.evt_engine import EVTEngine` 做 EVT VaR 约束
- [ ] 集成 `from jingsuan.risk_budget import RiskBudgetEngine` 做动态仓位限制
- [ ] 添加换手率约束、基数约束

### 3. 市场冲击模型 (zhixing/market_impact.py — 新建)
- [ ] Almgren-Chriss 永久+瞬时冲击
- [ ] 平方根流动性模型
- [ ] TWAP/VWAP 执行调度 stub
- [ ] 与 MockBroker 集成

### 4. 真实 live_broker (zhixing/live_broker.py — 新建)
- [ ] 定义实盘券商接口（AbstractBroker）
- [ ] 包含: submit_order, cancel_order, get_positions, get_account
- [ ] Stub 实现 + 接口文档

## 参考
- 蓝图: `lingshulianghuasheji/重构蓝图.md` 第 4 节
- 契约: `lingshulianghuasheji/CONTRACTS.md`
- 精算接口: `jingsuan/` (可 import, 不可修改)

## 质量门禁
- ruff check + pytest tests/test_juece/ tests/test_huice/ tests/test_zhixing/
- 不修改 jingsuan/ → 合并零冲突
