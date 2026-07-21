# CLAUDE.md — 灵枢量化系统

> **项目**: 灵枢 (LingShu) — 基于LLM多智能体与图神经网络的A股量化投资系统
> **版本**: v4.2.4 | 日期: 2026-07-21
> **设计文档**: (待重建)

---

## 一、项目架构

```
E:\28721\lingshu\
├── shuju/                  # 数据层 (14 py)
│   — AKShare + Tushare + 新闻/舆情采集
│   — 数据预处理/对齐/缓存 + 幸存者偏差治理
│   — 企业行为调整 (corporate_action) + 质量监控 (quality_monitor)
│
├── yinzi/                  # 量化因子引擎 (18 py)
│   — 41 因子 6 大类 (价值/动量/质量/波动率/情绪/另类)
│   — 卡尔曼滤波 + FDR 多重检验 + GARCH/HMM
│
├── zhinengti/              # LLM多智能体 ★ 核心创新 (13 py)
│   — 5 Agent 并行分析 (宏观/赛道/个股RAG/舆情/风控)
│   — LangChain 编排 + 结构化输出 + Bühlmann-Straub 融合
│
├── tushenjing/             # 图神经网络引擎 ★ 核心创新 (9 py)
│   — 5,208 节点 × 94,770 边 A 股异构图
│   — GCN/GAT/HGT + PyG 训练 + NumPy 推理
│
├── jingsuan/               # 精算中台 ★ 核心创新 (14 py)
│   — EVT 极值理论 (GPD/POT + GEV Block Maxima)
│   — Copula 连接函数 (含 DCC-GARCH 时变)
│   — 破产理论 / 信度理论 / 准备金评估
│   — VaR 回测 (Kupiec/Christoffersen/DQ/Berkowitz)
│   — 纯计算层: 无状态、无 IO、无数据库
│
├── juece/                  # 集成决策引擎 (10 py)
│   — Black-Litterman + HRP 组合优化
│   — 三路信号 Bühlmann-Straub 信度加权融合
│   — BL 协方差 Ledoit-Wolf 收缩 + 贝叶斯更新
│
├── fengkong/               # 风控模块 (9 py)
│   — 三态熔断器 + 仓位/行业/单票限制
│   — VaR/CVaR 实时计算 + 历史场景压力测试
│
├── zhixing/                # 交易执行 (8 py)
│   — 订单生命周期 + 批量调仓 + 模拟券商
│   — 成交记录 + 持仓追踪
│
├── huice/                  # 回测系统 (9 py)
│   — 事件驱动回测引擎 + 绩效归因
│   — DSR/PSR 防数据窥探 + Walk-Forward CV + 参数网格搜索
│
├── jiekou/                 # API 层 (6 py + routes/)
│   — FastAPI + WebSocket + 鉴权/限流/CORS
│   — 选股/智能体/持仓/风控/回测/交易/GNN 端点
│
├── shujuku/                # 持久化层 (8 py + models/)
│   — SQLAlchemy 2.0 ORM + Alembic 迁移
│   — Redis 缓存 + Prometheus 指标导出
│
├── qianduan/               # 前端 (28 源文件)
│   ├── src/
│   │   ├── pages/              # Dashboard / 选股 / 智能体 / 风控 / 回测 / 设置 (6 页面)
│   │   ├── components/         # 布局 / 仪表盘 / 图表 / 选股 / 智能体 / 通用 / 回测
│   │   ├── hooks/              # WebSocket / 行情 / 智能体 / 持仓 / 风控
│   │   ├── stores/             # Zustand 状态管理
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
│   ├── Dockerfile.frontend
│   ├── prometheus.yml
│   ├── grafana/dashboards.json
│   ├── nginx.conf
│   └── requirements-ci.txt
│
├── alembic/                # 数据库迁移
│   ├── env.py
│   ├── alembic.ini → ../alembic.ini
│   └── versions/
│
├── tests/                  # 测试 (按模块组织, 723 tests)
│   ├── test_shuju/            (9 test files)
│   ├── test_yinzi/            (6 test files)
│   ├── test_zhinengti/        (3 test files)
│   ├── test_tushenjing/       (3 test files)
│   ├── test_juece/            (5 test files)
│   ├── test_fengkong/         (1 test file)
│   ├── test_zhixing/          (2 test files)
│   ├── test_huice/            (3 test files)
│   ├── test_jiekou/           (2 test files)
│   ├── test_shujuku/          (5 test files)
│   └── test_integration_real.py (真实数据集成测试)
│
├── scripts/                # 运维脚本 (19 py)
│   ├── verify_jingsuan_v41.py        # jingsuan 精算引擎验证
│   ├── run_daily_pipeline.py         # 每日自动流水线
│   ├── download_tushare_data.py      # 历史数据下载
│   ├── compute_factors_from_db.py    # 因子计算 (从 DB)
│   ├── factor_quality_assessment.py  # 因子质量评估
│   ├── run_full_period_backtest.py   # 全周期回测
│   └── ...                           # (其他 13 脚本)
│
├── lingshulianghuasheji/   # 设计文档 (待重建)
│
├── CLAUDE.md               # 本文件
├── config.yaml              # 统一配置
├── pyproject.toml           # 项目配置 + 依赖
├── README.md                # 项目说明
└── 使用指南.md              # 使用指南
```

---

## 二、模块依赖关系

```
jiekou → juece, zhinengti, fengkong, shuju, shujuku
zhixing → fengkong (light), juece
fengkong → jingsuan (EVT/Copula/VaR), juece (Signal 模型)
juece → yinzi, tushenjing, zhinengti, jingsuan (精算约束嵌入BL)
zhinengti → shuju, shujuku
tushenjing → shuju, yinzi
yinzi → shuju
jingsuan → (standalone — 纯数学计算层, 无外部依赖)
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
- **禁止使用 `sys.path.insert()` hack`

### 线程安全
- 核心引擎：`threading.RLock` + Event 驱动
- 数据采集：`Queue(2000)` 解耦生产者/消费者
- API：`asyncio.Lock` 保护 WebSocket 连接

---

## 四、技术栈

| 层级 | 技术 | 理由 |
|------|------|------|
| 后端语言 | Python 3.10+ | AI/数据科学生态 |
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
  2. yinzi      → 41 因子并行计算 (30秒内)
  3. zhinengti  → 5 Agent 并行分析 → 结构化洞见
  4. tushenjing → 产业链图更新 + 信息传播 + 推理
  5. yinzi      → 卡尔曼滤波更新因子权重
  6. jingsuan   → EVT VaR + Copula + 信度融合 + 随机情景
  7. juece      → 三路信号加权融合 → 全市场排序 (精算约束嵌入 BL)
  8. juece      → Top 30 选股 + 风控过滤 + 调仓清单
  9. zhixing    → 模拟执行 → 成交记录
  10. shujuku   → 因子/信号/交易/智能体报告 持久化
  11. jiekou    → WebSocket 推送前端实时更新

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
| 1 | `shujuku` (数据库) | ✅ 完成 | 65 passed | 2026-06-02 |
| 2 | `shuju` (数据) | ✅ 完成 | 92 passed | 2026-06-03 |
| 3 | `yinzi` (因子) | ✅ 完成 | 94 passed | 2026-06-03 |
| 4 | `zhinengti` (智能体) | ✅ 完成 | 32 passed | 2026-06-03 |
| 5 | `tushenjing` (图神经) | ✅ 完成 | 67 passed | 2026-06-07 |
| 6 | `jingsuan` (精算中台) | ✅ 完成 | 53 passed | 2026-07-07 |
| 7 | `juece` (决策) | ✅ 完成 | 37 passed | 2026-06-10 |
| 8 | `fengkong` (风控) | ✅ 完成 | 26 passed | 2026-06-07 |
| 9 | `zhixing` (执行) | ✅ 完成 | 20 passed | 2026-06-07 |
| 10 | `huice` (回测) | ✅ 完成 | 20 passed | 2026-06-07 |
| 11 | `jiekou` (API) | ✅ 完成 | 19 passed | 2026-06-08 |
| 12 | `qianduan` (前端) | ✅ 完成 | build passed | 2026-06-15 |
| 13 | `bushu` (部署) | ✅ 完成 | — | 2026-06-03 |
