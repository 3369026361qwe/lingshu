<h1 align="center">灵枢 LingShu</h1>

<p align="center">
  <strong>基于 LLM 多智能体与图神经网络的 A 股量化投资系统</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-3.2.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/tests-444_passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## 📖 目录

- [核心创新](#核心创新)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [快速启动](#快速启动)
- [项目结构](#项目结构)
- [模块详解](#模块详解)
- [数据流](#数据流)
- [前端展示](#前端展示)
- [API 端点](#api-端点)
- [测试](#测试)
- [部署](#部署)
- [论文研究](#论文研究)
- [开发状态](#开发状态)
- [许可证](#许可证)

---

## 核心创新

| 创新点 | 说明 | 技术实现 |
|------|------|------|
| 🧠 **LLM 多智能体协作** | 5 个专业 Agent 并行分析市场，各司其职 | LangChain + DeepSeek API，结构化输出 |
| 🕸️ **GNN 产业链图** | 5,208 节点 × 94,770 条边，捕捉行业关联溢出效应 | PyG GAT 4-head，Top-5 精度 **87.8%** |
| 📐 **卡尔曼动态权重** | 因子权重随 IC 漂移在线自适应 | filterpy 卡尔曼滤波，11 个活跃因子 |
| 🔗 **三路信号融合** | 量化因子 + GNN + LLM Agent 独立路径 → 加权融合 | 卡尔曼权重 + IC 驱动动态分配 |
| 🛡️ **四层风控体系** | 熔断器 + 仓位限制 + VaR/CVaR + AI 风控 Agent | 实时监测，三态熔断 |
| 🎯 **端到端可视化** | 7 页面 PWA 前端，选股→交易→回测全链路展示 | React 18 + TypeScript + ECharts + Zustand |

---

## 系统架构

```
                    ┌─────────────────────────────────┐
                    │           qianduan (前端)         │
                    │   React + TS + ECharts + Zustand  │
                    └──────────────┬──────────────────┘
                                   │ HTTP / WebSocket
                    ┌──────────────▼──────────────────┐
                    │           jiekou (API)            │
                    │   FastAPI + PWA + 鉴权 + 限流     │
                    └──────┬───────┬───────┬──────────┘
                           │       │       │
              ┌────────────▼┐ ┌────▼──┐ ┌──▼──────────┐
              │   juece     │ │zhineng│ │  fengkong    │
              │  (决策引擎)  │ │  ti   │ │  (风控)      │
              │ 三路融合+选股│ │(智能体)│ │ 熔断+仓位+VaR│
              └──┬───┬───┬─┘ └──┬───┬┘ └──────┬───────┘
                 │   │   │      │   │         │
    ┌────────────▼┐ ┌▼───▼──┐ ┌▼───▼─────┐   │
    │   yinzi     │ │tushen │ │ shuju     │   │
    │  (因子引擎)  │ │ jing  │ │ (数据层)  │   │
    │ 35因子+卡尔曼│ │(GNN)  │ │ AKShare+  │   │
    └──────┬──────┘ └───┬───┘ │ Tushare   │   │
           │            │     └─────┬─────┘   │
           └────────────┴───────────┘         │
                        │                     │
                 ┌──────▼──────┐    ┌─────────▼────┐
                 │   shujuku   │    │   zhixing     │
                 │  (持久化)    │    │  (交易执行)    │
                 │ SQLAlchemy  │    │ 模拟/实盘券商  │
                 └─────────────┘    └──────────────┘
```

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.10+ / TypeScript 5.5+ | 后端 + 前端 |
| Web | FastAPI + Uvicorn + WebSocket | 异步 API 服务 |
| 数据库 | SQLAlchemy 2.0 + Alembic | ORM + 迁移 |
| GNN | PyTorch 2.5 + PyG 2.7 | 图神经网络训练/推理 |
| LLM | LangChain + DeepSeek API | 多智能体分析 |
| 数值 | NumPy + SciPy + Pandas + TA-Lib | 因子计算 |
| 滤波 | filterpy | 卡尔曼动态权重 |
| 前端 | React 18 + ECharts + Zustand + Ant Design | 暗色金融主题 PWA |
| 部署 | Docker Compose + Nginx + Prometheus | 容器化运维 |
| 测试 | pytest 444 / vitest 37 | 全栈覆盖 |

---

## 快速启动

### 前提条件

- Python ≥3.10，Node.js ≥18
- （可选）DeepSeek API Key（LLM 智能体，无 Key 则自动降级为规则模式）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/3369026361qwe/lingshu.git
cd lingshu

# 2. 安装 Python 依赖
pip install -e .

# 3. 构建前端
cd qianduan && npm install && npm run build && cd ..

# 4. 配置环境变量
cp .env.example .env   # 编辑 .env 填入 API Key（可选）
```

### 启动

```bash
# 一键启动（API + 前端）
uvicorn jiekou.server:app --host 0.0.0.0 --port 8000

# 访问：http://localhost:8000
```

### 开发模式

```bash
# 后端（热更新）
uvicorn jiekou.server:app --host 0.0.0.0 --port 8000 --reload

# 前端（热更新，另一终端）
cd qianduan && npm run dev
# 访问：http://localhost:5173
```

---

## 项目结构

```
lingshu/
├── shuju/              # 数据层 — AKShare + Tushare 数据获取与预处理
├── yinzi/              # 量化因子 — 35 因子 + 卡尔曼滤波动态权重
├── zhinengti/          # LLM 智能体 — 5 Agent 并行分析（核心创新）
├── tushenjing/         # GNN 引擎 — 产业链图 + GCN/GAT + NumPy 推理
├── juece/              # 决策引擎 — 三路信号融合 + 选股 + 组合优化
├── fengkong/           # 风控模块 — 熔断器 + 仓位限制 + VaR/CVaR
├── zhixing/            # 交易执行 — 订单管理 + 模拟券商
├── huice/              # 回测系统 — 事件驱动引擎 + 绩效归因
├── jiekou/             # API 层 — FastAPI + WebSocket + 中间件
│   └── routes/         #   REST 路由（9 个）
├── shujuku/            # 持久化 — SQLAlchemy ORM + Alembic 迁移
│   └── models/         #   12 个数据模型
├── qianduan/           # 前端 — React 18 + ECharts + PWA
│   └── src/
│       ├── pages/      #   7 个页面
│       ├── components/ #   4 个组件
│       ├── hooks/      #   3 个自定义 Hook
│       ├── stores/     #   3 个 Zustand Store
│       └── utils/      #   工具函数 + 类型定义
├── tests/              # 测试 — 按模块组织，444 tests
├── scripts/            # 运维脚本 — 数据下载 + 因子计算 + 回测运行
├── bushu/              # 部署 — Docker Compose + Prometheus + Grafana
├── alembic/            # 数据库迁移文件
├── data/               # 数据文件（.gitignore 保护）
│   ├── lingshu.db      #   主数据库（SQLite）
│   ├── gnn_model.pt    #   GAT 训练权重 (1.6MB)
│   └── gnn_config.json #   模型配置
└── lingshulianghuasheji/  # 设计文档 + 优化计划 + 论文方案
```

---

## 模块详解

### `shuju/` — 数据层（7 文件 · 92 tests）
多源数据获取与预处理：AKShare（免费行情）、Tushare Pro（财务/估值）、新闻舆情采集。支持数据对齐、缺失值处理、行业中性化。

### `yinzi/` — 量化因子引擎（10 文件 · 55 tests）
35 个量化因子并行计算：价值/动量/质量/波动率/情绪/另类 6 大类。卡尔曼滤波在线估计因子权重，IC/IR 检验驱动动态分配。活跃因子 11 个。

### `zhinengti/` — LLM 多智能体（11 文件 · 18 tests）
5 个专业 Agent 并行分析：宏观分析师、赛道分析师、个股分析师（含 RAG）、舆情分析师、风险监控 Agent。通过 Orchestrator 调度，支持 DeepSeek/OpenAI/本地 Qwen。

### `tushenjing/` — 图神经网络（6 文件 · 35 tests）
A 股产业链图构建（5208 节点 × 94770 边）。GCN/GAT 双架构，PyG 训练 + 纯 NumPy 推理路径。GAT 4-head 训练结果：Top-5 精度 87.8%。

### `juece/` — 决策引擎（6 文件 · 22 tests）
三路信号加权融合：量化因子 + GNN + LLM Agent。Black-Litterman 组合优化 + 调仓计算。LambdaRank 排序学习。

### `jiekou/` — API 层（9 文件 · 12 tests）
FastAPI + WebSocket。9 个路由模块、API Key 鉴权、限流中间件。服务前端静态文件，支持 PWA 安装。

### `qianduan/` — 前端（28 源文件 · 37 tests）
7 页面暗色金融主题 PWA：仪表盘、选股、智能体、风控、交易、回测、GNN 产业链图。TypeScript 全覆盖，React.lazy 懒加载，入口 12.8KB。

---

## 数据流

```
每日15:00收盘后（自动/手动触发）:
  1. shuju      → 拉取全市场日线 + 财报 + 新闻
  2. yinzi      → 35 因子并行计算（30 秒内）
  3. zhinengti  → 5 Agent 并行分析 → 结构化洞见
  4. tushenjing → 产业链图更新 + GNN 推理
  5. yinzi      → 卡尔曼滤波更新因子权重
  6. juece      → 三路信号加权融合 → 全市场排序
  7. juece      → Top-30 选股 + 组合优化 + 风控过滤
  8. zhixing    → 模拟执行 → 成交记录
  9. shujuku    → 因子/信号/交易/智能体报告持久化
  10. jiekou    → WebSocket 推送前端实时更新
```

---

## 前端展示

| 页面 | 路由 | 功能 | 图表 |
|------|------|------|------|
| 🏠 仪表盘 | `/` | 三指数行情 + AI 洞察 + 选股 | 权益曲线 + 因子权重 |
| 📈 选股 | `/selection` | Top-N 得分排名 | 得分分布柱状图 |
| 🤖 智能体 | `/agents` | 5 Agent 详细报告 | 信号分布饼图 |
| ⚠️ 风控 | `/risk` | 风险状态 + 持仓 | VaR 仪表盘 + 仓位饼图 |
| 💰 交易 | `/trading` | 选股→组合→调仓→执行 | 组合配置饼图 + 风险仪表 |
| 🧪 回测 | `/backtest` | 回测配置 + 绩效 | 权益曲线 |
| 🕸️ GNN | (Dashboard) | 产业链力导向图 | 可拖拽缩放 |

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/selection` | 选股 Top-N |
| GET | `/api/agents/reports` | Agent 报告 |
| GET | `/api/portfolio` | 当前持仓 |
| GET | `/api/equity` | 权益曲线 |
| GET | `/api/risk/status` | 风控状态 |
| GET | `/api/factors/weights` | 因子权重 |
| GET | `/api/gnn/graph` | GNN 产业链图 |
| GET | `/api/trade/pipeline` | 完整交易链路 |
| POST | `/api/trade/execute` | 执行交易 |
| GET | `/api/trade/history` | 交易历史 |
| POST | `/api/backtest` | 运行回测 |
| GET | `/api/backtest/summary` | 回测绩效 |
| WS | `/ws/market` | 市场实时推送 |
| WS | `/ws/agents` | Agent 实时推送 |
| WS | `/ws/risk` | 风控实时推送 |

---

## 测试

```bash
# 全量测试
python -m pytest tests/ -v              # 后端 444 tests
cd qianduan && npx vitest run           # 前端 37 tests

# 按模块
python -m pytest tests/test_yinzi/ -v   # 因子
python -m pytest tests/test_huice/ -v   # 回测
python -m pytest tests/test_jiekou/ -v  # API
```

| 模块 | 测试数 | 状态 |
|------|:--:|:--:|
| shujuku | 60 | ✅ |
| shuju | 92 | ✅ |
| yinzi | 55 | ✅ |
| zhinengti | 18 | ✅ |
| tushenjing | 35 | ✅ |
| juece | 22 | ✅ |
| fengkong | 23 | ✅ |
| zhixing | 11 | ✅ |
| huice | 14 | ✅ |
| jiekou | 12 | ✅ |
| qianduan | 37 | ✅ |
| **合计** | **444 + 37** | **全通过** |

---

## 部署

```bash
cd bushu
docker-compose up -d
```

服务架构：
- **api**: FastAPI + 前端静态文件（端口 8000）
- **db**: PostgreSQL（可选，默认 SQLite）
- **redis**: 缓存（可选）
- **prometheus**: 指标采集
- **grafana**: 仪表盘（端口 3000）

所有服务均配置 healthcheck。

---

## 论文研究

本项目可作为**金融工程本科/硕士论文**的研究平台。核心学术贡献：

1. **LLM 多智能体与量化因子融合** — 5 Agent 结构化分析输出 + 卡尔曼动态加权
2. **GNN 产业链信息传播** — 5208 节点异构图，证明行业关联包含独立于价量因子的增量信息
3. **三路信号集成决策** — 因子 + GNN + Agent 的消融实验框架

详见 [`lingshulianghuasheji/论文优化方案.md`](lingshulianghuasheji/论文优化方案.md)

---

## 开发状态

| 顺序 | 模块 | 状态 | 测试 |
|:--:|------|:--:|:--:|
| 1 | shujuku (数据库) | ✅ | 60 |
| 2 | shuju (数据) | ✅ | 92 |
| 3 | yinzi (因子) | ✅ | 55 |
| 4 | zhinengti (智能体) | ✅ | 18 |
| 5 | tushenjing (图神经) | ✅ | 35 |
| 6 | juece (决策) | ✅ | 22 |
| 7 | fengkong (风控) | ✅ | 23 |
| 8 | zhixing (执行) | ✅ | 11 |
| 9 | huice (回测) | ✅ | 14 |
| 10 | jiekou (API) | ✅ | 12 |
| 11 | qianduan (前端) | ✅ | 37 |
| 12 | bushu (部署) | ✅ | — |

---

## 许可证

MIT License

---

<p align="center">
  <sub>Built with ❤️ by 灵枢团队 | 2026</sub>
</p>
