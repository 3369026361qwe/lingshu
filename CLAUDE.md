# CLAUDE.md — 数据工程师窗口 (灵枢 v4.0 feat/shuju-enhance)

## 角色: 数据工程师

你负责完善 `shuju/` 数据管道和 `shujuku/` 数据模型，解决幸存者偏差、企业行为、数据质量三大问题。

## 约束
- **修改/新增 `shuju/` 和 `shujuku/` 目录内文件**
- 不修改其他模块（即使看到可改进之处）
- 现有数据获取接口不变（akshare_fetcher, tushare_fetcher 不改）
- 每个新模块必须有独立验证

## 任务清单

### 1. 幸存者偏差治理 (shuju/universe_manager.py — 升级)
- [ ] 从 AKShare 获取退市股票列表，构建 `data/delisted_stocks.csv`
- [ ] 维护 {code: (list_date, delist_date)} 数据库
- [ ] `survivorship_free_universe(date)` 返回完整的 Point-in-Time 股票宇宙
- [ ] 添加 ST/*ST 实时过滤（从 AKShare stock_info 中读取 `name` 字段）
- [ ] 添加停牌检测（当日成交量=0 且无涨跌停标记）

### 2. 企业行为处理 (shuju/corporate_action.py — 升级)
- [ ] 从 AKShare 获取复权因子数据
- [ ] `build_adjustment_factors()` 从 real data 构建复权因子序列
- [ ] 支持前复权和后复权两种模式
- [ ] 验证: 复权后的收益率与不复权收益率的差异分析

### 3. 数据质量监控 (shuju/quality_monitor.py — 升级)
- [ ] 连接 `DataAligner` 输出 → 自动检查缺失率
- [ ] 每日数据拉取后自动运行质量检查
- [ ] 异常检测: 单日数据量 < 历史均值的 50% → 告警
- [ ] 生成每日数据质量报告 (JSON 或 log)

### 4. PostgreSQL 迁移准备 (shujuku/)
- [ ] 新增 `alembic/versions/` 中的 PostgreSQL 迁移脚本
- [ ] `shujuku/config.py` 中 DATABASE_URL 支持 PostgreSQL 连接池配置
- [ ] 保持 SQLite 开发模式不变（`DB_HOST` 未设置时自动使用 SQLite）

### 5. 数据缓存优化
- [ ] 检查 `shuju/cache_manager.py` 是否正确使用 `shuju.config.get_config()`
- [ ] 验证 Redis 降级逻辑（Redis 不可用 → 内存缓存）
- [ ] 添加缓存命中率监控

## 参考
- 蓝图: `lingshulianghuasheji/重构蓝图.md` 第 7 节
- 契约: `lingshulianghuasheji/CONTRACTS.md`
- 现有 stub: `shuju/universe_manager.py` `shuju/corporate_action.py` `shuju/quality_monitor.py`

## 质量门禁
- ruff check + pytest tests/test_shuju/ tests/test_shujuku/
- 不修改 jingsuan/ juece/ yinzi/ → 合并零冲突
