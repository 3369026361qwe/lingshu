# CLAUDE.md — 精算师窗口 (灵枢 v4.2 feat/jingsuan-enhance)

## 角色: 精算师

你负责深化 `jingsuan/` 精算中台。当前已有 14 个引擎模块，从专业精算师视角进行深度优化。

## 约束
- **只修改/新增 `jingsuan/` 目录内的文件**
- 不修改任何其他模块（即使看到可改进之处）
- 所有接口严格遵守 `lingshulianghuasheji/CONTRACTS.md` 中的契约
- 所有金融计算使用 `Decimal` 类型
- 从 `shuju.utils` 导入 `safe_divide, safe_mean`
- 每个引擎必须有独立的数学正确性验证（合成数据测试）

## 已完成任务

### 1. EVT 引擎深化 (evt_engine.py)
- [x] 实现完整的 MLE GPD 拟合（Newton-Raphson 迭代）
- [x] 添加 Profile Likelihood 置信区间
- [x] 实现自动阈值选择（mean excess plot + 稳定性准则）
- [x] 添加 GEV 分布拟合（Block Maxima 方法作为备选）
- [x] 修复 PWM 参数估计（生存权重修正）

### 2. Copula 引擎深化 (copula_engine.py)
- [x] 实现 DCC (Dynamic Conditional Correlation) 时变 Copula → dcc_copula.py
- [x] 添加 Rotated Gumbel (180°) 用于下尾依赖建模
- [x] 实现多维 Copula (Vine Copula 分解)
- [x] 添加 Cramér-von Mises GoF 检验 (Bootstrap p-value)
- [x] 添加 Gaussian Copula
- [x] 修复 simulate() 精确条件采样 (Archimedean h-function + Cholesky)
- [x] 修复二元正态 CDF (12点 Gauss-Legendre)

### 3. 破产理论深化 (ruin_engine.py)
- [x] 实现 Cramér-Lundberg 精确解（复合Poisson+指数索赔）
- [x] 添加 Beekman-Bowers 近似
- [x] 实现多期破产概率的条件更新 (Bayesian update)

### 4. 信度理论深化 (credibility.py)
- [x] 实现分层信度模型 (Hierarchical Credibility — 体制内+体制间)
- [x] 添加回归信度 (Hachemeister 模型)
- [x] 实现信度因子的时序衰减加权
- [x] 修复 exp overflow 防护

### 5. 新增模块
- [x] `jingsuan/dcc_copula.py` — DCC-GARCH Copula 时变依赖
- [x] `jingsuan/gev_engine.py` — GEV Block Maxima + Return Level
- [x] `jingsuan/reserving.py` — Chain Ladder / Bornhuetter-Ferguson / Cape Cod

### 6. VaR 回测深化 (var_backtest.py)
- [x] 添加 Dynamic Quantile (DQ) 检验 (Engle-Manganelli 2004)
- [x] 实现 Berkowitz 似然比检验 (PIT → N(0,1) → AR(1) LR)
- [x] 添加 Traffic Light 滚动窗口评估
- [x] 修复 Christoffersen LR_cc = LR_uc + LR_ind
- [x] 修复 chi2_sf (正确级数/续分数实现)

## 质量状态
- ruff: CLEAN (E/F/W/I/N/UP/B)
- 合成数据验证: 53/53 passed
- 文件: 14 .py, 5,208 lines
- 审查: 3 轮深度审查, 所有发现已修复
- 垃圾文件: 已清理 (__pycache__, .pyc)

## 参考
- 蓝图文档: `lingshulianghuasheji/重构蓝图.md` 第 3 节
- 接口契约: `lingshulianghuasheji/CONTRACTS.md`
- 测试: `tests/verify_jingsuan_v41.py`
