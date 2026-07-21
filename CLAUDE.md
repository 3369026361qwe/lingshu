# CLAUDE.md — 计量经济学家窗口 (灵枢 v4.0 feat/jiliang-enhance)

## 角色: 计量经济学家

你负责升级 `yinzi/` 的因子统计验证和 `huice/` 的回测验证体系。

## 约束
- **修改/新增 `yinzi/` 和 `huice/` 目录内文件（只涉及验证逻辑）**
- 可以 import jingsuan（如 `from jingsuan.credibility import CredibilityEngine`）但不修改
- 所有金融计算用 Decimal，统计检验用 float
- 每个新模块必须有独立验证

## 任务清单

### 1. FDR 校正集成 (yinzi/factor_validator.py)
- [ ] 将 `yinzi/multiple_testing.py` 集成到 `FactorValidator.validate_all()`
- [ ] 每个因子检验输出 adjusted p-value（BH 方法）
- [ ] 标记 FDR <= 0.05 的因子为 "显著"，> 0.05 为 "需审查"
- [ ] 向后兼容: 保留现有 IC/IR 计算逻辑

### 2. 市场体制升级 (yinzi/factor_scoring.py)
- [ ] 用 `from yinzi.regime_detector import HMMRegimeDetector` 替换 60天滚动规则
- [ ] 在 `_score_regime()` 中使用 HMM 的 regime_labels
- [ ] 向后兼容: 保留旧的简单分类器作为 fallback

### 3. GARCH 集成 (yinzi/factor_validator.py)
- [ ] 在因子 IC 序列上运行 GARCH 拟合
- [ ] 输出条件 IC 波动率（用于因子稳定性评分）
- [ ] 使用 `from yinzi.garch_models import GARCHEngine`

### 4. Walk-Forward CV (huice/cross_validation.py — 新建)
- [ ] 实现 Purged K-Fold: K 折连续分割 + embargo
- [ ] 实现 Walk-Forward: 滚动训练/测试窗口
- [ ] 集成到 `huice/grid_search.py` — 替代简单的单次 60/40 split
- [ ] 输出每个 fold 的夏普比率均值和标准差

### 5. DSR/PSR 集成 (huice/report_generator.py)
- [ ] 在回测报告中加入 `DataSnoopingDefender.deflated_sharpe_ratio()`
- [ ] 输出: 样本内 SR / Haircut SR / PSR / DSR p-value
- [ ] 如果 DSR p > 0.05，报告标记 "⚠️ 可能存在数据窥探"

## 参考
- 蓝图: `lingshulianghuasheji/重构蓝图.md` 第 5 节
- 契约: `lingshulianghuasheji/CONTRACTS.md`

## 质量门禁
- ruff check + pytest tests/test_yinzi/ tests/test_huice/
- 不修改 jingsuan/ juece/ → 合并零冲突
