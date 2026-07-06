"""
市场冲击模型 (v4.0).

Almgren-Chriss 永久+瞬时冲击 + 平方根流动性模型 + TWAP/VWAP 执行调度 stub.

数学基础:
    永久冲击 (信息泄漏): g(v) = γ · σ · v / V
    瞬时冲击 (流动性消耗): h(v) = η · σ · (v / V)^(3/5)
    总冲击成本: C(v) = g(v) + h(v)
    平方根模型: impact(bps) = κ · sqrt(v / V)

其中:
    v = 订单量 (股数), V = 日均成交量, σ = 日波动率
    γ = 永久冲击系数 (~0.1), η = 瞬时冲击系数 (~0.15), κ ~ 10-50 bps

Usage:
    from zhixing.market_impact import MarketImpactModel, ImpactEstimate
    impact = MarketImpactModel.estimate(order_quantity, daily_volume, volatility)
"""

from dataclasses import dataclass
from decimal import Decimal

from shuju.utils import safe_divide


@dataclass
class ImpactEstimate:
    """市场冲击估计."""
    permanent_impact: Decimal     # 永久冲击 (bps, 1 bps = 0.0001)
    temporary_impact: Decimal     # 瞬时冲击 (bps)
    total_impact: Decimal         # 总冲击成本 (bps)
    total_cost: Decimal           # 总冲击成本 (金额)
    arrival_price: Decimal        # 到达价格
    effective_price: Decimal      # 有效执行价格
    participation_rate: Decimal   # 参与率 (v/V)


class MarketImpactModel:
    """市场冲击模型 — Almgren-Chriss 框架 + 平方根流动性模型."""

    # 默认参数 (基于实证文献)
    DEFAULT_GAMMA = Decimal("0.10")   # 永久冲击系数
    DEFAULT_ETA = Decimal("0.15")     # 瞬时冲击系数
    DEFAULT_KAPPA = Decimal("20")     # 平方根模型系数 (bps)

    @staticmethod
    def almgren_chriss(
        order_quantity: int,
        daily_volume: int,
        daily_volatility: Decimal,
        price: Decimal,
        gamma: Decimal | None = None,
        eta: Decimal | None = None,
    ) -> ImpactEstimate:
        """Almgren-Chriss 冲击模型.

        Args:
            order_quantity: 订单股数
            daily_volume: 日均成交量 (股)
            daily_volatility: 日波动率 (如 0.02 = 2%)
            price: 当前价格
            gamma: 永久冲击系数 (默认 0.10)
            eta: 瞬时冲击系数 (默认 0.15)

        Returns:
            ImpactEstimate with costs in both bps and absolute terms
        """
        g = gamma if gamma is not None else MarketImpactModel.DEFAULT_GAMMA
        e = eta if eta is not None else MarketImpactModel.DEFAULT_ETA

        v = Decimal(str(order_quantity))
        V = Decimal(str(daily_volume))
        sigma = daily_volatility

        participation = safe_divide(v, V)

        # 永久冲击: g(v) = γ · σ · v / V  (以价格比例表示)
        permanent = g * sigma * participation

        # 瞬时冲击: h(v) = η · σ · (v/V)^(3/5)
        temp_exponent = participation ** Decimal("0.6")  # 3/5 = 0.6
        temporary = e * sigma * temp_exponent

        # 总冲击 (比例)
        total_ratio = permanent + temporary

        # 转换为 bps
        permanent_bps = permanent * Decimal("10000")
        temporary_bps = temporary * Decimal("10000")
        total_bps = total_ratio * Decimal("10000")

        # 成本金额
        total_cost = total_ratio * price * v

        # 有效执行价格
        effective_price = price * (Decimal("1") + total_ratio)

        return ImpactEstimate(
            permanent_impact=permanent_bps,
            temporary_impact=temporary_bps,
            total_impact=total_bps,
            total_cost=total_cost,
            arrival_price=price,
            effective_price=effective_price,
            participation_rate=participation,
        )

    @staticmethod
    def sqrt_liquidity(
        order_quantity: int,
        daily_volume: int,
        price: Decimal,
        kappa: Decimal | None = None,
    ) -> ImpactEstimate:
        """平方根流动性冲击模型.

        impact(bps) = κ · sqrt(v / V)

        这是更简单的行业标准模型。
        适用于流动性较好的中大盘股。

        Args:
            order_quantity: 订单股数
            daily_volume: 日均成交量 (股)
            price: 当前价格
            kappa: 冲击系数 bps (默认 20, 范围 10-50)

        Returns:
            ImpactEstimate (永久冲击=0, 全部归为瞬时)
        """
        k = kappa if kappa is not None else MarketImpactModel.DEFAULT_KAPPA

        v = Decimal(str(order_quantity))
        V = Decimal(str(daily_volume))
        participation = safe_divide(v, V)

        # impact(bps) = κ · sqrt(v/V)
        total_bps = k * participation.sqrt()

        total_ratio = total_bps / Decimal("10000")
        total_cost = total_ratio * price * v
        effective_price = price * (Decimal("1") + total_ratio)

        return ImpactEstimate(
            permanent_impact=Decimal("0"),
            temporary_impact=total_bps,
            total_impact=total_bps,
            total_cost=total_cost,
            arrival_price=price,
            effective_price=effective_price,
            participation_rate=participation,
        )

    @staticmethod
    def estimate(
        order_quantity: int,
        daily_volume: int,
        daily_volatility: Decimal,
        price: Decimal,
        model: str = "almgren_chriss",
    ) -> ImpactEstimate:
        """统一接口: 估计订单的市场冲击.

        Args:
            order_quantity: 订单股数
            daily_volume: 日均成交量
            daily_volatility: 日波动率
            price: 当前价格
            model: "almgren_chriss" | "sqrt_liquidity"

        Returns:
            ImpactEstimate
        """
        if model == "sqrt_liquidity":
            return MarketImpactModel.sqrt_liquidity(order_quantity, daily_volume, price)
        return MarketImpactModel.almgren_chriss(
            order_quantity, daily_volume, daily_volatility, price
        )

    @staticmethod
    def optimal_twap_slices(
        order_quantity: int,
        daily_volume: int,
        daily_volatility: Decimal,
        price: Decimal,
        trading_minutes: int = 240,
        slice_interval_minutes: int = 15,
    ) -> list[dict]:
        """TWAP 执行调度 stub.

        将总订单切分为等时间间隔的子订单。
        返回每片的理想执行时间、数量和预期冲击。

        Args:
            order_quantity: 总订单股数
            daily_volume: 日均成交量
            daily_volatility: 日波动率
            price: 当前价格
            trading_minutes: 总交易分钟 (默认 240 = 4小时)
            slice_interval_minutes: 每片间隔分钟 (默认 15)

        Returns:
            切片列表: [{slice_start, quantity, expected_impact_bps, ...}]
        """
        n_slices = max(1, trading_minutes // slice_interval_minutes)
        qty_per_slice = max(100, order_quantity // n_slices // 100 * 100)  # A股按手取整

        # 调整以匹配总数量
        remaining = order_quantity - qty_per_slice * (n_slices - 1)
        if remaining < 100:
            remaining = qty_per_slice
            n_slices = max(1, order_quantity // qty_per_slice)

        slices = []
        for i in range(n_slices):
            qty = qty_per_slice if i < n_slices - 1 else remaining
            slice_vol = daily_volume // n_slices

            impact = MarketImpactModel.sqrt_liquidity(qty, max(slice_vol, 1), price)

            slices.append({
                "slice_index": i + 1,
                "slice_start_minute": i * slice_interval_minutes,
                "quantity": qty,
                "participation_rate": float(safe_divide(Decimal(str(qty)), Decimal(str(max(slice_vol, 1))))),
                "expected_impact_bps": float(impact.total_impact),
                "expected_cost": float(impact.total_cost),
            })

        return slices

    @staticmethod
    def optimal_vwap_slices(
        order_quantity: int,
        daily_volume: int,
        daily_volatility: Decimal,
        price: Decimal,
        volume_profile: list[float] | None = None,
    ) -> list[dict]:
        """VWAP 执行调度 stub.

        按历史成交量分布比例分配子订单。
        成交量大的时段分配更多订单，减少冲击。

        Args:
            order_quantity: 总订单股数
            daily_volume: 日均成交量
            daily_volatility: 日波动率
            price: 当前价格
            volume_profile: 日内成交量分布 (如 24 个 15-分钟时段的权重),
                           默认均匀分布

        Returns:
            切片列表
        """
        n_slices = 16  # 默认 16 个 15-分钟时段
        if volume_profile is None:
            volume_profile = [1.0 / n_slices] * n_slices

        # 归一化
        total_w = sum(volume_profile)
        norm_weights = [w / total_w for w in volume_profile]

        slices = []
        allocated = 0
        for i, w in enumerate(norm_weights):
            if i == n_slices - 1:
                qty = order_quantity - allocated
            else:
                qty = max(100, int(order_quantity * w) // 100 * 100)
            allocated += qty

            slice_vol = int(daily_volume * w)
            impact = MarketImpactModel.sqrt_liquidity(qty, max(slice_vol, 1), price)

            slices.append({
                "slice_index": i + 1,
                "slice_start_minute": i * 15,
                "volume_weight": w,
                "quantity": qty,
                "expected_impact_bps": float(impact.total_impact),
                "expected_cost": float(impact.total_cost),
            })

        return slices
