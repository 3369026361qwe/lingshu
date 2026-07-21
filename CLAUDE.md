# CLAUDE.md — 软件架构师窗口 (灵枢 v4.0 feat/jiagou-enhance)

## 角色: 软件架构师

你负责全局代码质量整理。不新增功能，只做清理、统一、优化。

## 约束
- 可修改任何文件的导入、格式、代码风格
- **不改变任何函数的输入/输出行为**（只做等价重构）
- 每完成一个子模块跑 ruff check + pytest 验证不破坏已有测试

## 任务清单

### 1. 全局配置统一
- [x] 确保 `from shujuku.settings import load_config` 被各模块使用（替换硬编码默认值）
- [x] 依次检查 `fengkong/` `juece/` `huice/` `zhixing/` 中的硬编码参数 → 从 config.yaml 读取
- [x] `shuju/constants.py` — 0 imports, 已安全删除 (所有配置已迁移至 shuju/config.py)

### 2. 10 个 metrics.py 整理
- [x] 不改现有 metrics.py 的内容（功能和名称不变）
- [x] 只在 `shujuku/metrics.py` 加桥接注释：`# 统一入口。各模块的独立 metrics.py 在 Phase 5 迁移` — 已验证存在，Phase 编号已修正为 Phase 5
- [x] 确保 `from shujuku.metrics import new_counter, new_histogram, new_gauge` 可用 — 已验证
- [x] 审计: 10 个 metrics.py, 79 唯一指标名, 0 重复, 383 行

### 3. 死代码清理
- [x] `shuju/constants.py` — 确认 0 imports → 已安全删除
- [x] 检查 `shuju/config.py` 是否被正确使用（所有 TTL/常量引用） — 4 文件导入，配置完整
- [x] grep 全项目找 `# TODO` `# FIXME` `# HACK` → 0 结果，项目清洁

### 4. pyproject.toml 优化
- [x] 添加 ruff 严格规则: F841, F401, B011 — 已被 F/B 选择器覆盖，无需额外添加
- [x] 版本号确认: pyproject.toml="4.0.0", config.yaml="4.0.0", package.json="4.0.0", App.tsx="4.0.0" — 全部一致
- [x] 确认 jingsuan/ 在 setuptools find 范围内 — 未被排除，正确包含

### 5. 导入规范检查
- [x] 确保 0 个 `sys.path.insert` 调用 — 仅 `alembic/env.py:14` 保留（Alembic 标准写法）
- [x] 模块内相对导入 `from .xxx`，跨模块绝对导入 `from yinzi.xxx` — 已验证一致
- [x] ruff I001 (import sorting) 全项目零违规 — 已验证 ✓

### 6. 文档同步
- [x] README.md vs 实际文件数对照 — 测试计数已更新为实际值
- [x] CLAUDE.md 更新开发状态表（v4.0 已完成模块） — 所有任务标记完毕

## 完成状态

**feat/jiagou-enhance — 全部 6 项任务已完成** (2026-07-21)

| 任务 | 变更 |
|------|------|
| 全局配置统一 | `shuju/constants.py` 删除, 0 imports |
| Metrics 整理 | 桥接注释 Phase 4→5, 10 文件 79 指标 0 重复 |
| 死代码清理 | 0 TODO/FIXME/HACK |
| pyproject.toml | F841/F401/B011 已覆盖, 版本一致, jingsuan 已包含 |
| 导入规范 | 0 sys.path(except alembic), I001 全通过 |
| 文档同步 | README 测试数更新, CLAUDE.md 完成标记 |

## 参考
- 蓝图: `lingshulianghuasheji/重构蓝图.md` 第 6 节
- 契约: `lingshulianghuasheji/CONTRACTS.md`

## 质量门禁
- `ruff check . --quiet` 全项目零违规
- `python -m pytest tests/ -x` 全量通过
- 不改变任何已有函数行为
