# CLAUDE.md — 灵枢量化系统

> **项目**: 灵枢 (LingShu) — 基于LLM多智能体与图神经网络的A股量化投资系统
> **版本**: v2.0 | 日期: 2026-06-02
> **设计文档**: lingshulianghuasheji/灵枢量化系统-设计文档.md

---

## 一、项目架构

```
E:\28721\lingshu\
├── shuju/                  # 数据层 (7 py)
│   ├── akshare_fetcher.py     # AKShare 行情数据
│   ├── tushare_fetcher.py     # Tushare 财务/估值
│   ├── news_fetcher.py        # 新闻/公告
│   ├── sentiment_fetcher.py   # 社交媒体舆情
│   ├── data_preprocessor.py   # 去极值/标准化/行业中性化
│   ├── data_aligner.py        # 多源数据时间对齐
│   └── cache_manager.py       # 数据缓存
│
├── yinzi/                  # 量化因子引擎 (10 py)
│   ├── factor_base.py         # 因子抽象基类
│   ├── value_factors.py       # PE/PB/PS/FCFYield/PEG
│   ├── momentum_factors.py    # 1M/3M/6M/12-1M 动量
│   ├── quality_factors.py     # ROE/ROA/毛利率/净利率
│   ├── volatility_factors.py  # 历史/下行/Beta/VaR
│   ├── sentiment_factors.py   # 成交量/资金流向/北向
│   ├── alternative_factors.py # 分析师/机构/股东
│   ├── kalman_weight.py       # 卡尔曼滤波动态权重
│   ├── factor_validator.py    # IC/IR/分层回测检验
│   └── factor_store.py        # 因子持久化
│
├── zhinengti/              # LLM多智能体 ★ 核心创新 (11 py)
│   ├── agent_base.py          # Agent 基类 (工具注册/记忆/输出)
│   ├── orchestrator.py        # 调度协调器 (任务分配/聚合)
│   ├── macro_analyst.py       # 宏观分析师 Agent
│   ├── sector_analyst.py      # 赛道分析师 Agent
│   ├── stock_analyst.py       # 个股分析师 Agent (RAG)
│   ├── sentiment_analyst.py   # 舆情分析师 Agent
│   ├── risk_monitor.py        # 风险监控 Agent
│   ├── agent_tools.py         # Agent 工具集
│   ├── llm_client.py          # LLM 调用客户端
│   ├── rag_pipeline.py        # RAG 检索增强生成
│   └── prompt_templates.py    # 系统提示词模板
│
├── tushenjing/             # 图神经网络引擎 ★ 核心创新 (6 py)
│   ├── graph_builder.py       # A股产业链图构建
│   ├── gnn_model.py           # GNN 模型定义 (GCN/GAT/HGT)
│   ├── graph_trainer.py       # 模型训练与验证
│   ├── graph_inference.py     # 推理与特征传播
│   ├── graph_updater.py       # 图结构动态更新
│   └── graph_utils.py         # 图数据处理工具
│
├── juece/                  # 集成决策引擎 (5 py)
│   ├── ensemble_engine.py     # 三路信号加权融合
│   ├── stock_selector.py      # 选股信号生成器
│   ├── portfolio_optimizer.py # Black-Litterman 组合优化
│   ├── rebalancer.py          # 调仓计算 (目标 vs 当前)
│   └── benchmark.py           # 基准比较 (沪深300/中证500)
│
├── fengkong/               # 风控模块 (8 py)
│   ├── circuit_breaker.py     # 三态熔断器
│   ├── rate_limiter.py        # 调仓频率限制
│   ├── position_limiter.py    # 仓位/行业/单票限制
│   ├── position_tracker.py    # 实时持仓追踪
│   ├── var_calculator.py      # VaR/CVaR 实时计算
│   ├── stress_tester.py       # 历史场景压力测试
│   ├── risk_manager.py        # 风控总入口
│   └── risk_models.py         # 风控数据模型
│
├── zhixing/                # 交易执行 (5 py)
│   ├── order_manager.py       # 订单生命周期
│   ├── batch_executor.py      # 批量调仓执行
│   ├── mock_broker.py         # 模拟券商
│   ├── live_broker.py         # 实盘券商接口 (未来)
│   └── trade_recorder.py      # 成交记录器
│
├── huice/                  # 回测系统 (5 py)
│   ├── backtest_engine.py     # 事件驱动回测引擎
│   ├── performance_metrics.py # 夏普/回撤/IC/IR/换手率
│   ├── attribution.py         # 绩效归因分析
│   ├── grid_search.py         # 参数网格搜索
│   └── report_generator.py    # 回测报告生成
│
├── jiekou/                 # API 层 (7 py)
│   ├── server.py              # FastAPI 入口 + WebSocket
│   ├── routes/
│   │   ├── selection_routes.py   # 选股端点
│   │   ├── agent_routes.py       # 智能体报告端点
│   │   ├── portfolio_routes.py   # 持仓管理端点
│   │   ├── risk_routes.py        # 风险监控端点
│   │   └── huice_routes.py       # 回测控制端点
│   ├── schemas.py             # Pydantic V2 模型
│   ├── dependencies.py        # 依赖注入
│   └── middleware.py          # 中间件 (日志/限流/CORS)
│
├── shujuku/                # 持久化层 (8 py)
│   ├── config.py              # 数据库配置
│   ├── models/
│   │   ├── market_models.py      # 行情/财务 ORM
│   │   ├── yinzi_models.py       # 因子/权重 ORM
│   │   ├── zhinengti_models.py   # 智能体报告 ORM
│   │   ├── jiaoyi_models.py      # 交易/持仓 ORM
│   │   └── fengkong_models.py    # 风控记录 ORM
│   ├── repository.py          # CRUD 统一仓库
│   ├── session.py             # 会话管理
│   ├── redis_cache.py         # Redis 缓存
│   └── metrics.py             # Prometheus 指标导出
│
├── qianduan/               # 前端 (28 源文件)
│   ├── src/
│   │   ├── pages/              # Dashboard / 选股 / 智能体 / 风控 / 回测 / 设置
│   │   ├── components/         # 布局 / 仪表盘 / 图表 / 选股 / 智能体 / 通用 / 回测
│   │   ├── hooks/              # 5 自定义 hooks (WebSocket/行情/智能体/持仓/风控)
│   │   ├── i18n/               # 中英文双语 (zh-CN / en)
│   │   ├── styles/             # 暗色金融主题 + 动效
│   │   ├── types/              # API 类型定义
│   │   └── utils/              # 格式化 + 常量
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── bushu/                  # 部署运维
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.qianduan
│   ├── prometheus.yml
│   ├── grafana/dashboards.json
│   ├── nginx.conf
│   └── init.sql
│
├── alembic/                # 数据库迁移
│   ├── env.py
│   ├── alembic.ini
│   └── versions/
│
├── tests/                  # 测试 (按模块组织)
│   ├── test_shuju/
│   ├── test_yinzi/
│   ├── test_zhinengti/
│   ├── test_tushenjing/
│   ├── test_juece/
│   ├── test_fengkong/
│   ├── test_zhixing/
│   ├── test_jiekou/
│   └── test_huice/
│
├── scripts/                # 运维脚本
│   ├── init_shujuku.py        # 数据库初始化
│   ├── download_historical.py # 历史数据下载
│   └── run_huice.py           # 回测运行
│
├── lingshulianghuasheji/   # 设计文档
│   ├── 灵枢量化系统-设计文档.md
│   └── 灵枢量化系统-设计演示.pptx
│
├── CLAUDE.md               # 本文件
├── pyproject.toml           # 项目配置 + 依赖
└── README.md                # 项目说明
```

---

## 二、模块依赖关系

```
jiekou → juece, zhinengti, fengkong, shuju, shujuku
zhixing → fengkong (light), juece
fengkong → juece (Signal 模型)
juece → yinzi, tushenjing, zhinengti
zhinengti → shuju, shujuku
tushenjing → shuju, yinzi
yinzi → shuju
shujuku → (standalone, SQLAlchemy)
huice → shuju, yinzi, juece, zhixing
qianduan → jiekou (HTTP/WebSocket)
bushu → (standalone, Docker)
```

---

## 三、核心约定

### 精度要求
- 所有金融计算使用 `Decimal` 类型，精度 18 位，严禁 `float`
- API 金额字段使用 `str` 序列化以避免 JSON 精度损失
- 使用 `safe_divide()` 而非 `/` 运算符防止除零

### 导入约定
- **模块内部使用相对导入**：`from .models import ...`
- **跨模块使用绝对导入**：`from yinzi.factor_base import FactorBase`
- **禁止使用 `sys.path.insert()` hack**

### 线程安全
- 核心引擎：`threading.RLock` + Event 驱动
- 数据采集：`Queue(2000)` 解耦生产者/消费者
- API：`asyncio.Lock` 保护 WebSocket 连接

---

## 四、技术栈

| 层级 | 技术 | 理由 |
|------|------|------|
| 后端语言 | Python 3.11+ | AI/数据科学生态 |
| 后端框架 | FastAPI + Uvicorn | 异步高性能 + 原生WebSocket |
| LLM框架 | LangChain + 自建Agent | Agent编排 + 金融场景定制 |
| LLM模型 | 本地 Qwen3-32B / API deepseek-v4 | 中文金融理解力强 |
| GNN | PyTorch Geometric (PyG) | 最成熟GNN库 |
| 数值计算 | NumPy + SciPy + Pandas | 科学计算标配 |
| 技术指标 | TA-Lib | 80+ 金融技术指标 |
| 卡尔曼滤波 | filterpy | 轻量无额外依赖 |
| 数据获取 | AKShare + Tushare Pro | 免费全覆盖 |
| NLP | jieba + transformers | 中文分词 + 情感分析 |
| 前端 | React 18 + TypeScript + Vite | 现代前端工程化 |
| UI | Ant Design 5 + ECharts | 企业级组件 + 金融图表 |
| 数据库 | PostgreSQL (生产) / SQLite (开发) | 关系型 + JSON 灵活存储 |
| 缓存 | Redis | 因子缓存 + Agent结果缓存 |
| 容器化 | Docker Compose | 一键部署 |
| 监控 | Prometheus + Grafana | 指标采集 + 可视化 |

---

## 五、运行方式

```bash
# 启动 API 服务 (开发模式)
cd E:\28721\lingshu
uvicorn jiekou.server:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
uvicorn jiekou.server:app --host 0.0.0.0 --port 8000

# 自定义 LLM 后端
LLM_BACKEND=deepseek LLM_API_KEY=xxx uvicorn jiekou.server:app --port 8000

# 运行测试
python -m pytest tests/ -v

# 运行单模块测试
python -m pytest tests/test_yinzi/ -v

# Docker 部署
cd bushu && docker-compose up -d
```

---

## 六、数据流

```
每日 15:00 收盘后:
  1. shuju      → 拉取全市场日线 + 财报 + 新闻舆情
  2. yinzi      → 25+ 因子并行计算 (30秒内)
  3. zhinengti  → 5 Agent 并行分析 → 结构化洞见
  4. tushenjing → 产业链图更新 + 信息传播 + 推理
  5. yinzi      → 卡尔曼滤波更新因子权重
  6. juece      → 三路信号加权融合 → 全市场排序
  7. juece      → Top 30 选股 + 风控过滤 + 调仓清单
  8. zhixing    → 模拟执行 → 成交记录
  9. shujuku    → 因子/信号/交易/智能体报告 持久化
  10. jiekou    → WebSocket 推送前端实时更新

实时模式 (9:30-15:00):
  - 每 5 分钟拉取快照数据
  - zhinengti 舆情 Agent 持续监控异常信号
  - fengkong 风险 Agent 实时评估持仓风险
  - 前端每 10 秒推送市场概览更新
```

---

## 七、开发状态

| 顺序 | 模块 | 状态 | 测试 | 完成日期 |
|:--:|------|:--:|:--:|:--:|
| 1 | `shujuku` (数据库) | ✅ 完成 | 60 passed | 2026-06-02 |
| 2 | `shuju` (数据) | ✅ 完成 | 92 passed | 2026-06-03 |
| 3 | `yinzi` (因子) | ✅ 完成 | 55 passed | 2026-06-03 |
| 4 | `zhinengti` (智能体) | ✅ 完成 | 18 passed | 2026-06-03 |
| 5 | `tushenjing` (图神经) | ✅ 完成 | 35 passed | 2026-06-07 |
| 6 | `juece` (决策) | ✅ 完成 | 22 passed | 2026-06-10 |
| 7 | `fengkong` (风控) | ✅ 完成 | 23 passed | 2026-06-07 |
| 8 | `zhixing` (执行) | ✅ 完成 | 11 passed | 2026-06-07 |
| 9 | `huice` (回测) | ✅ 完成 | 14 passed | 2026-06-07 |
| 10 | `jiekou` (API) | ✅ 完成 | 12 passed | 2026-06-08 |
| 11 | `qianduan` (前端) | ✅ 完成 | build passed | 2026-06-15 |
| 12 | `bushu` (部署) | ✅ 完成 | — | 2026-06-03 |

