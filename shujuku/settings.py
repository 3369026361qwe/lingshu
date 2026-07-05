"""
应用配置加载器 (v4.0)。

从 config.yaml 读取全局配置，支持环境变量模板替换 (${VAR})，
以 dataclass 形式提供类型安全的配置访问。

Usage:
    from shujuku.settings import load_config
    config = load_config()
    print(config.fusion.weights.factor)
"""

import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import yaml

# ── ENV 模板替换 ────────────────────────────────────────────────
_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _subst_env(value: str) -> str:
    """替换字符串中的 ${VAR} 模板为环境变量值。"""
    def _replace(m: re.Match) -> str:
        return os.getenv(m.group(1), "")
    return _ENV_PATTERN.sub(_replace, value)


# ── 配置数据类 ──────────────────────────────────────────────────


@dataclass
class DataConfig:
    tushare_token: str = ""
    data_dir: str = "data/"


@dataclass
class FactorConfig:
    enabled: list[str] = field(default_factory=list)
    min_valid: int = 8


@dataclass
class GNNConfig:
    features: int = 20
    hidden_dim: int = 64
    dropout: float = 0.3
    epochs: int = 100
    learning_rate: float = 0.005
    forward_days: int = 20
    loss: str = "pairwise_ranking"
    margin: float = 0.02


@dataclass
class ModelConfig:
    gnn: GNNConfig = field(default_factory=GNNConfig)


@dataclass
class FusionWeights:
    factor: Decimal = Decimal("0.33")
    gnn: Decimal = Decimal("0.33")
    agent: Decimal = Decimal("0.34")

    def __post_init__(self) -> None:
        if isinstance(self.factor, float):
            self.factor = Decimal(str(self.factor))
        if isinstance(self.gnn, float):
            self.gnn = Decimal(str(self.gnn))
        if isinstance(self.agent, float):
            self.agent = Decimal(str(self.agent))

    def normalize(self) -> "FusionWeights":
        """归一化到总和为 1，处理零权重场景。"""
        total = self.factor + self.gnn + self.agent
        if total == 0:
            n = Decimal("1") / Decimal("3")
            return FusionWeights(factor=n, gnn=n, agent=n)
        return FusionWeights(
            factor=self.factor / total,
            gnn=self.gnn / total,
            agent=self.agent / total,
        )


@dataclass
class FusionConfig:
    weights: FusionWeights = field(default_factory=FusionWeights)


@dataclass
class BacktestConfig:
    top_n: int = 40
    rebalance_days: int = 40
    initial_capital: int = 1_000_000
    commission: Decimal = Decimal("0.0003")
    slippage: Decimal = Decimal("0.001")
    benchmark: str = "hs800_equal_weight"

    def __post_init__(self) -> None:
        if isinstance(self.commission, float):
            self.commission = Decimal(str(self.commission))
        if isinstance(self.slippage, float):
            self.slippage = Decimal(str(self.slippage))


@dataclass
class RiskConfig:
    max_drawdown_pct: Decimal = Decimal("0.15")
    single_stock_max_weight: Decimal = Decimal("0.10")
    single_industry_max_weight: Decimal = Decimal("0.30")
    var_confidence: Decimal = Decimal("0.95")
    max_consecutive_losses: int = 5

    def __post_init__(self) -> None:
        for name in ("max_drawdown_pct", "single_stock_max_weight",
                     "single_industry_max_weight", "var_confidence"):
            val = getattr(self, name)
            if isinstance(val, float):
                setattr(self, name, Decimal(str(val)))


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/lingshu.log"


@dataclass
class AppConfig:
    """应用全局配置 (v4.0)。"""
    version: str = "4.0.0"
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ── 加载器 ──────────────────────────────────────────────────────


class ConfigLoader:
    """配置单例加载器 — 首次调用时从 config.yaml 加载，后续返回缓存。"""

    _instance: AppConfig | None = None

    @classmethod
    def load(cls, path: str = "config.yaml") -> AppConfig:
        """加载配置，支持环境变量模板替换。"""
        if cls._instance is not None:
            return cls._instance

        raw_text = Path(path).read_text(encoding="utf-8")
        raw_text = _subst_env(raw_text)
        raw: dict = yaml.safe_load(raw_text)

        cls._instance = cls._parse(raw)
        return cls._instance

    @classmethod
    def _parse(cls, raw: dict) -> AppConfig:
        raw_data = raw.get("data", {})
        raw_factors = raw.get("factors", {})
        raw_model = raw.get("model", {})
        raw_gnn = raw_model.get("gnn", {})
        raw_fusion = raw.get("fusion", {})
        raw_fw = raw_fusion.get("weights", {})
        raw_bt = raw.get("backtest", {})
        raw_risk = raw.get("risk", {})
        raw_api = raw.get("api", {})
        raw_log = raw.get("logging", {})

        return AppConfig(
            version=raw.get("version", "4.0.0"),
            data=DataConfig(
                tushare_token=raw_data.get("tushare_token", ""),
                data_dir=raw_data.get("data_dir", "data/"),
            ),
            factors=FactorConfig(
                enabled=list(raw_factors.get("enabled", [])),
                min_valid=int(raw_factors.get("min_valid", 8)),
            ),
            model=ModelConfig(
                gnn=GNNConfig(
                    features=int(raw_gnn.get("features", 20)),
                    hidden_dim=int(raw_gnn.get("hidden_dim", 64)),
                    dropout=float(raw_gnn.get("dropout", 0.3)),
                    epochs=int(raw_gnn.get("epochs", 100)),
                    learning_rate=float(raw_gnn.get("learning_rate", 0.005)),
                    forward_days=int(raw_gnn.get("forward_days", 20)),
                    loss=str(raw_gnn.get("loss", "pairwise_ranking")),
                    margin=float(raw_gnn.get("margin", 0.02)),
                )
            ),
            fusion=FusionConfig(
                weights=FusionWeights(
                    factor=Decimal(str(raw_fw.get("factor", 0.33))),
                    gnn=Decimal(str(raw_fw.get("gnn", 0.33))),
                    agent=Decimal(str(raw_fw.get("agent", 0.34))),
                )
            ),
            backtest=BacktestConfig(
                top_n=int(raw_bt.get("top_n", 40)),
                rebalance_days=int(raw_bt.get("rebalance_days", 40)),
                initial_capital=int(raw_bt.get("initial_capital", 1_000_000)),
                commission=Decimal(str(raw_bt.get("commission", 0.0003))),
                slippage=Decimal(str(raw_bt.get("slippage", 0.001))),
                benchmark=str(raw_bt.get("benchmark", "hs800_equal_weight")),
            ),
            risk=RiskConfig(
                max_drawdown_pct=Decimal(str(raw_risk.get("max_drawdown_pct", 0.15))),
                single_stock_max_weight=Decimal(str(raw_risk.get("single_stock_max_weight", 0.10))),
                single_industry_max_weight=Decimal(str(raw_risk.get("single_industry_max_weight", 0.30))),
                var_confidence=Decimal(str(raw_risk.get("var_confidence", 0.95))),
                max_consecutive_losses=int(raw_risk.get("max_consecutive_losses", 5)),
            ),
            api=APIConfig(
                host=str(raw_api.get("host", "0.0.0.0")),
                port=int(raw_api.get("port", 8000)),
            ),
            logging=LoggingConfig(
                level=str(raw_log.get("level", "INFO")),
                file=str(raw_log.get("file", "logs/lingshu.log")),
            ),
        )

    @classmethod
    def reset(cls) -> None:
        """清除缓存（测试用）。"""
        cls._instance = None


# ── 便捷函数 ────────────────────────────────────────────────────


def load_config(path: str = "config.yaml") -> AppConfig:
    """加载应用配置（单例缓存）。"""
    return ConfigLoader.load(path)
