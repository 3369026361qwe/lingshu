<h1 align="center">灵枢 LingShu v4.0</h1>

<p align="center">
  <strong>精算级 A 股量化投资系统 — LLM 多智能体 × 图神经网络 × 精算风控</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/tests-480_passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## 目录

- [核心创新](#核心创新)
- [系统架构](#系统架构)
- [快速启动](#快速启动)
- [项目结构](#项目结构)
- [五大角色重构蓝图](#五大角色重构蓝图)
- [模块详解](#模块详解)
- [数据流](#数据流)
- [API 端点](#api-端点)
- [测试](#测试)
- [部署](#部署)
- [论文研究](#论文研究)
- [许可证](#许可证)

---

## 核心创新

| 创新点 | 说明 | 技术实现 |
|------|------|------|
| 🧠 **LLM 多智能体** | 5 专业 Agent 并行分析：宏观/赛道/个股(RAG)/舆情/风控 | LangChain + DeepSeek，结构化输出 |
| 🕸️ **GNN 产业链图** | 5,208 节点 × 94,770 边，捕捉行业关联溢出效应 | PyG GAT 4-head，Top-5 精度 **87.8%** |
| 📐 **精算中台** | EVT 尾部风险 + Copula 依赖 + 破产理论 + 信度融合 | `jingsuan/` 纯计算引擎，无状态无 IO |
| 🔗 **三路信号融合** | 量化因子 + GNN + LLM Agent → Bühlmann-Straub 信度加权 | 替代简单线性 IC 加权 |
| 🛡️ **五层风控** | 精算中台 + 熔断器 + 仓位限制 + VaR回测 + 实时监控 | Kupiec/Christoffersen/Acerbi-Szekely 检验 |
| 🎯 **端到端可视化** | 7 页面 PWA 前端，选股→交易→回测全链路 | React 18 + TypeScript + ECharts + Zustand |

---

## 系统架构

```
┌─────────────────────────────────────────────┐
│                qianduan/ (前端)               │
│        React 18 + TS + ECharts + Zustand      │
├─────────────────────────────────────────────┤
│                jiekou/ (API)                  │
│        FastAPI + WebSocket + 鉴权 + 限流      │
├─────────────────────────────────────────────┤
│    juece/ (决策)          huice/ (回测验证)    │
│    zhixing/ (执行)        fengkong/ (风控)     │
├─────────────────────────────────────────────┤
│          jingsuan/ (精算中台) ★ v4.0 新增      │
│   EVT │ Copula │ 破产理论 │ 信度理论 │ 随机过程  │
├─────────────────────────────────────────────┤
│  yinzi/ (因子) │ zhinengti/ (智能体) │ tushenjing/ (GNN) │
├─────────────────────────────────────────────┤
│              shuju/ (数据层)                   │
│    AKShare + Tushare + 幸存者偏差治理          │
├─────────────────────────────────────────────┤
│  shujuku/ (SQLAlchemy) │ bushu/ (Docker)       │
└─────────────────────────────────────────────┘

横切层: 统一配置加载 | 统一指标导出 | 统一日志追踪
```

**关键架构升级 (v3 → v4)**：新增 `jingsuan/` 精算中台作为计算引擎层，将风控从「事后过滤器」升级为「内生约束」嵌入组合优化；闭环反馈架构替代线性流水线。

---

## 快速启动

### 前提

- Python ≥3.11, Node.js ≥18
- (可选) DeepSeek API Key — LLM 智能体使用

### 安装

```bash
git clone https://github.com/3369026361qwe/lingshu.git
cd lingshu
pip install -e .
cd qianduan && npm install && npm run build && cd ..
cp .env.example .env
```

### 启动

```bash
uvicorn jiekou.server:app --host 0.0.0.0 --port 8000
# → http://localhost:8000
```

### 开发

```bash
# 后端热更新
uvicorn jiekou.server:app --host 0.0.0.0 --port 8000 --reload
# 前端热更新 (另一终端)
cd qianduan && npm run dev  # → http://localhost:5173
```

---

## 项目结构

```
lingshu/
├── shuju/              # 数据层 — AKShare + Tushare + 幸存者偏差治理
├── yinzi/              # 量化因子 — 41 因子 + 卡尔曼滤波 + FDR 多重检验校正
├── zhinengti/          # LLM 智能体 — 5 Agent 并行分析
├── tushenjing/         # GNN 引擎 — 异构图 GCN/GAT + 体制感知传播
├── jingsuan/           # ★ 精算中台 (v4.0 新增)
│   ├── evt_engine.py       # EVT 极值理论 — GPD/POT 尾部 VaR
│   ├── copula_engine.py    # Copula 连接函数 — 多资产尾部依赖
│   ├── ruin_engine.py      # 破产理论 — Lundberg 动态风险预算
│   ├── credibility.py      # 信度理论 — Bühlmann-Straub 信号融合
│   ├── solvency.py         # Solvency II SCR — 风险资本聚合
│   ├── scenario_gen.py     # 随机情景生成器
│   ├── var_backtest.py     # VaR 回测 — Kupiec/Christoffersen/Acerbi
│   ├── stress_engine.py    # 反向压力测试 + 协方差情景生成
│   └── risk_budget.py      # 动态风险预算引擎
├── juece/              # 决策引擎 — BL + HRP 组合优化
├── fengkong/           # 风控执行 — 熔断器 + 仓位限制 + 实时监控
├── zhixing/            # 交易执行 — 订单管理 + 市场冲击模型
├── huice/              # 回测系统 — 事件驱动 + 归因分析 + DSR/PSR
├── jiekou/             # API 层 — FastAPI + WebSocket + 中间件
│   └── routes/         #   REST 路由 (9 个)
├── shujuku/            # 持久化 — SQLAlchemy 2.0 + Alembic
│   └── models/         #   18 个 ORM 模型
├── qianduan/           # 前端 — React 18 + PWA
│   └── src/
│       ├── pages/      #   7 页面
│       ├── components/ #   组件库
│       ├── hooks/      #   自定义 Hooks
│       └── stores/     #   Zustand 状态管理
├── tests/              # 测试 — 按模块组织，480+ tests
├── scripts/            # 运维脚本 — 薄调用层
├── bushu/              # 部署 — Docker Compose + Prometheus + Grafana
├── alembic/            # 数据库迁移
├── data/               # 数据文件 (SQLite / GNN 权重 / GNN 预测)
└── lingshulianghuasheji/  # 设计文档 + 重构蓝图 + 论文方案
```

---

## 五大角色重构蓝图

灵枢 v4.0 通过五个专业角色的联合重构，实现「个人实盘盈利 + 机构尽调通过」双重目标。

| 角色 | 负责模块 | 核心方法论 |
|------|---------|-----------|
| 🧮 **精算师** | `jingsuan/` | EVT · Copula · 破产理论 · 信度理论 · Solvency II |
| 📊 **量化金工** | `juece/` `huice/` `zhixing/` | Black-Litterman · HRP · Almgren-Chriss · 归因分析 |
| 📈 **计量经济** | `yinzi/` `huice/` 验证 | FDR · DSR/PSR · GARCH · HMM · Walk-Forward CV |
| 🏗️ **软件架构** | 全局工程 | Decimal 统一 · Config 加载 · 脚本重构 · 死代码清理 |
| 🗄️ **数据工程** | `shuju/` `shujuku/` | 幸存者偏差 · 企业行为 · 数据质量 · PostgreSQL |

### 分阶段推进

```
Phase 0: 软件架构 + 数据工程  →  接口契约 + 基础设施
Phase 1: 精算师             →  jingsuan/ 精算中台
Phase 2: 量化金工            →  BL + HRP + 事件驱动回测 + 归因
Phase 3: 计量经济            →  FDR + DSR + GARCH + HMM
Phase 4: 软件架构            →  代码整理 + 统一化
Phase 5: 数据工程            →  数据治理 + PostgreSQL
```

### 并行开发（5 个 Claude Code 窗口 + git worktree）

每个角色在独立 git worktree 中工作，通过接口契约协同。详见 **[重构蓝图](lingshulianghuasheji/重构蓝图.md)** — 包含完整数学公式、接口契约、伪代码和验证方法论。

---

## 模块详解

### `shuju/` — 数据层
多源数据获取与预处理：AKShare（免费行情）、Tushare Pro（财务/估值）。数据对齐、去极值、标准化、行业中性化。**v4.0** 增加幸存者偏差治理、退市股票 Point-in-Time 宇宙、企业行为调整。

### `yinzi/` — 量化因子
41 个因子，6 大类：价值/动量/质量/波动率/情绪/另类。卡尔曼滤波在线估计因子权重。**v4.0** 增加 FDR 多重检验校正、GARCH 条件波动率、HMM 市场体制检测。

### `zhinengti/` — LLM 多智能体
5 个专业 Agent 并行：宏观/赛道/个股(RAG)/舆情/风控。Orchestrator 调度。**v4.0** 信度理论 Bühlmann-Straub 融合替代固定线性权重。

### `tushenjing/` — 图神经网络
5,208 节点 × 94,770 边 A 股异构图。GCN/GAT 双架构，PyG 训练 + NumPy 推理。GAT 4-head Top-5 精度 87.8%。

### `jingsuan/` — 精算中台 ★ v4.0
纯计算层，无状态、无 IO。EVT 极值理论 (GPD/POT) 替换正态 VaR；Copula 多资产尾部依赖替代线性 beta；破产理论动态风险预算替代固定仓位限制；信度理论信号融合。包含完整 VaR 回测检验套件。

### `juece/` — 决策引擎
Black-Litterman (Ledoit-Wolf 收缩 + 贝叶斯更新) + 分层风险平价 (HRP)。Almgren-Chriss 市场冲击模型。Brinson + 因子 + 风险三维归因分析。

### `huice/` — 回测系统
真正事件驱动回测引擎，集成 MockBroker + OrderManager。Deflated Sharpe Ratio + Probabilistic Sharpe Ratio 防数据窥探。Purged Walk-Forward 交叉验证。

---

## 数据流

```
每日15:00收盘后:
  1. shuju      → 拉取全市场日线 + 财报 + 新闻
  2. yinzi      → 41 因子并行计算 + FDR 筛选
  3. zhinengti  → 5 Agent 并行分析 → 结构化洞见
  4. tushenjing → 产业链图更新 + GNN 推理
  5. jingsuan   → EVT VaR + Copula 依赖 + 信度融合 + 动态风险预算
  6. juece      → BL/HRP 组合优化 (嵌入精算约束)
  7. fengkong   → 熔断器 + 仓位验证 + 实时监控
  8. zhixing    → 市场冲击模型 + 模拟执行
  9. huice      → 归因分析 + DSR/PSR 验证
  10. jiekou    → WebSocket 推送 → 前端实时更新
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/selection` | 选股 Top-N |
| GET | `/api/agents/reports` | Agent 报告 |
| GET | `/api/portfolio` | 当前持仓 |
| GET | `/api/equity` | 权益曲线 |
| GET | `/api/risk/status` | 风控状态 + VaR 回测 |
| GET | `/api/risk/var-backtest` | VaR 回测结果 (Kupiec/Christoffersen) |
| GET | `/api/gnn/graph` | GNN 产业链图 |
| GET/POST | `/api/trade/*` | 交易链路 |
| GET/POST | `/api/backtest/*` | 回测 + 绩效报告 + 归因 |
| WS | `/ws/market` `/ws/agents` `/ws/risk` | 实时推送 |

---

## 测试

```bash
python -m pytest tests/ -v              # 后端 480 tests
cd qianduan && npx vitest run           # 前端 37 tests

# 按模块
python -m pytest tests/test_yinzi/ -v
python -m pytest tests/test_jingsuan/ -v   # 精算中台
python -m pytest tests/test_huice/ -v
```

| 模块 | 测试数 | 状态 |
|------|:--:|:--:|
| shujuku | 65 | ✅ |
| shuju | 92 | ✅ |
| yinzi | 94 | ✅ |
| zhinengti | 32 | ✅ |
| tushenjing | 67 | ✅ |
| jingsuan | — | 🔜 v4.1 |
| juece | 37 | ✅ |
| fengkong | 26 | ✅ |
| zhixing | 20 | ✅ |
| huice | 20 | ✅ |
| jiekou | 19 | ✅ |
| qianduan | 37 | ✅ |
| **合计** | **480** | |

---

## 部署

```bash
cd bushu && docker-compose up -d
```

API (8000) + PostgreSQL + Redis + Prometheus + Grafana (3000)，全部 healthcheck。

---

## 论文研究

核心学术贡献：
1. **精算方法论在量化投资中的应用** — EVT/Copula/破产理论/信度理论系统性迁移
2. **LLM 多智能体与量化因子融合** — Bühlmann-Straub 信度加权框架
3. **GNN 产业链信息传播** — 体制感知异构图神经网络

详见 [`lingshulianghuasheji/论文优化方案.md`](lingshulianghuasheji/论文优化方案.md)

---

## 许可证

MIT License

---

<p align="center">
  <sub>Built with ❤️ by 灵枢团队 | 2026</sub>
</p>
