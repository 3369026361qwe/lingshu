"""
CRUD 统一仓库。

提供全部 ORM 模型的通用数据访问层，支持优雅降级：
- 数据库可用 → 正常读写
- 数据库不可用 → 返回默认值 / 空列表（不抛异常阻断上层）

Usage:
    repo = Repository(session)
    stocks = repo.get_active_stocks()
    repo.save_factor_value(code="000001", date=..., factor_name="pe", raw_value=15.5)
"""

import functools
import time
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Generator, List, Optional, Sequence, Type, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shujuku.models import Base, utcnow
from shujuku.models.market_models import DailyBar, FinancialReport, IndustryClassification, StockInfo
from shujuku.models.yinzi_models import FactorICRecord, FactorValue, FactorWeight
from shujuku.models.zhinengti_models import AgentEvidence, AgentReport
from shujuku.models.jiaoyi_models import Order, PortfolioSnapshot, Position, Trade
from shujuku.models.fengkong_models import CircuitBreakerEvent, RiskLog, VaRRecord
from shujuku.metrics import db_ops_total, db_ops_latency, db_errors_total, db_degraded

T = TypeVar("T", bound=Base)


def _track_db(operation: str, table: str):
    """装饰器：自动记录数据库操作的计数、延迟和错误。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            db_ops_total.labels(operation=operation, table=table).inc()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                db_ops_latency.labels(operation=operation, table=table).observe(
                    time.perf_counter() - start
                )
                return result
            except Exception as e:
                db_errors_total.labels(
                    operation=operation, error_type=type(e).__name__
                ).inc()
                raise
        return wrapper
    return decorator


class Repository:
    """统一数据仓库，优雅降级模式。"""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._degraded = False

    def _set_degraded(self) -> None:
        """标记降级模式并上报 Prometheus 指标。"""
        if not self._degraded:
            self._degraded = True
            db_degraded.set(1)

    def _clear_degraded(self) -> None:
        """LOW-12: 数据库恢复后清除降级标志。"""
        if self._degraded:
            self._degraded = False
            db_degraded.set(0)

    # ── 通用方法 ────────────────────────────────────────────

    @_track_db("insert", "generic")
    def add(self, obj: Base) -> None:
        """添加单条记录。"""
        self._session.add(obj)

    @_track_db("insert", "batch")
    def add_all(self, objects: Sequence[Base]) -> None:
        """批量添加记录。"""
        self._session.add_all(objects)

    def flush(self) -> None:
        """刷新到数据库（不提交）。"""
        self._session.flush()

    @_track_db("select", "generic")
    def get(self, model: Type[T], pk: Any) -> Optional[T]:
        """按主键查询。"""
        try:
            return self._session.get(model, pk)
        except Exception:
            self._set_degraded()
            return None

    @_track_db("select", "generic")
    def get_all(self, model: Type[T], limit: int = 1000) -> List[T]:
        """查询全部记录（带上限）。"""
        try:
            stmt = select(model).limit(limit)
            return list(self._session.scalars(stmt).all())
        except Exception:
            return []

    @_track_db("select", "generic")
    def count(self, model: Type[T]) -> int:
        """查询记录数。"""
        try:
            stmt = select(func.count()).select_from(model)
            return self._session.scalar(stmt) or 0
        except Exception:
            return 0

    # ── 股票信息 ────────────────────────────────────────────

    def get_active_stocks(self) -> List[StockInfo]:
        """获取所有活跃股票。"""
        try:
            stmt = select(StockInfo).where(StockInfo.is_active == True).order_by(StockInfo.code)  # noqa: E712
            return list(self._session.scalars(stmt).all())
        except Exception:
            return []

    def get_stock_by_code(self, code: str) -> Optional[StockInfo]:
        """按代码查询股票。"""
        return self.get(StockInfo, code)

    def upsert_stock(self, code: str, name: str, exchange: str = "SZ", listing_date: date | None = None) -> StockInfo:
        """插入或更新股票基础信息。"""
        try:
            existing = self.get(StockInfo, code)
            if existing:
                existing.name = name
                existing.exchange = exchange
                if listing_date:
                    existing.listing_date = listing_date
                existing.updated_at = utcnow()
                return existing
            stock = StockInfo(code=code, name=name, exchange=exchange, listing_date=listing_date)
            self.add(stock)
            return stock
        except Exception:
            self._set_degraded()
            return StockInfo(code=code, name=name, exchange=exchange, listing_date=listing_date)

    # ── 日线行情 ────────────────────────────────────────────

    def get_daily_bar(self, code: str, trade_date: date) -> Optional[DailyBar]:
        """查询单日行情。"""
        try:
            stmt = (
                select(DailyBar)
                .where(DailyBar.code == code)
                .where(DailyBar.trade_date == trade_date)
            )
            return self._session.scalar(stmt)
        except Exception:
            return None

    def get_daily_bars(self, code: str, start: date, end: date) -> List[DailyBar]:
        """查询日期范围内的日线数据。"""
        try:
            stmt = (
                select(DailyBar)
                .where(DailyBar.code == code)
                .where(DailyBar.trade_date >= start)
                .where(DailyBar.trade_date <= end)
                .order_by(DailyBar.trade_date)
            )
            return list(self._session.scalars(stmt).all())
        except Exception:
            return []

    def get_bars_for_date(self, trade_date: date) -> List[DailyBar]:
        """获取全市场某日行情。"""
        try:
            stmt = select(DailyBar).where(DailyBar.trade_date == trade_date)
            return list(self._session.scalars(stmt).all())
        except Exception:
            return []

    # ── 因子 ────────────────────────────────────────────────

    def save_factor_value(
        self,
        code: str,
        trade_date: date,
        category: str,
        factor_name: str,
        raw_value: Decimal,
        z_score: Decimal | None = None,
        percentile: Decimal | None = None,
    ) -> FactorValue:
        """保存单条因子值 (upsert)。"""
        try:
            stmt = (
                select(FactorValue)
                .where(FactorValue.code == code)
                .where(FactorValue.trade_date == trade_date)
                .where(FactorValue.factor_name == factor_name)
            )
            existing = self._session.scalar(stmt)
            if existing:
                existing.raw_value = raw_value
                existing.z_score = z_score
                existing.percentile = percentile
                existing.updated_at = utcnow()
                return existing

            # insert 也放在 try 内，保证原子性
            fv = FactorValue(
                code=code,
                trade_date=trade_date,
                category=category,
                factor_name=factor_name,
                raw_value=raw_value,
                z_score=z_score,
                percentile=percentile,
            )
            self.add(fv)
            return fv
        except Exception:
            self._set_degraded()
            # 返回未持久化对象，调用方自行判断
            return FactorValue(
                code=code, trade_date=trade_date,
                category=category, factor_name=factor_name,
                raw_value=raw_value,
                z_score=z_score,          # P1-5: 补全降级返回字段
                percentile=percentile,
            )

    def get_factor_values(self, code: str, factor_name: str, start: date, end: date) -> List[FactorValue]:
        """查询某股票某因子的历史值。"""
        try:
            stmt = (
                select(FactorValue)
                .where(FactorValue.code == code)
                .where(FactorValue.factor_name == factor_name)
                .where(FactorValue.trade_date >= start)
                .where(FactorValue.trade_date <= end)
                .order_by(FactorValue.trade_date)
            )
            return list(self._session.scalars(stmt).all())
        except Exception:
            return []

    def save_factor_weight(self, trade_date: date, factor_name: str, weight: Decimal, variance: Decimal | None = None) -> FactorWeight:
        """保存因子权重 (upsert)。"""
        try:
            stmt = (
                select(FactorWeight)
                .where(FactorWeight.trade_date == trade_date)
                .where(FactorWeight.factor_name == factor_name)
            )
            existing = self._session.scalar(stmt)
            if existing:
                existing.weight = weight
                existing.variance = variance
                existing.updated_at = utcnow()
                return existing

            fw = FactorWeight(trade_date=trade_date, category="", factor_name=factor_name, weight=weight, variance=variance)
            self.add(fw)
            return fw
        except Exception:
            self._set_degraded()
            return FactorWeight(trade_date=trade_date, factor_name=factor_name, weight=weight, variance=variance)

    # ── 智能体报告 ──────────────────────────────────────────

    def save_agent_report(
        self,
        agent_id: str,
        analysis_date: datetime,
        target_stocks: str,
        signal: Decimal,
        confidence: Decimal,
        reasoning: str,
        risk_flags: str | None = None,
        model_used: str | None = None,
        tokens_used: int | None = None,
        latency_ms: int | None = None,
        is_cached: bool = False,
    ) -> AgentReport:
        """保存 Agent 分析报告。"""
        report = AgentReport(
            agent_id=agent_id,
            analysis_date=analysis_date,
            target_stocks=target_stocks,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            risk_flags=risk_flags,
            model_used=model_used,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            is_cached=is_cached,
        )
        self.add(report)
        return report

    def get_latest_agent_reports(self, agent_id: str | None = None, limit: int = 10) -> List[AgentReport]:
        """获取最新的 Agent 报告。"""
        try:
            stmt = select(AgentReport).order_by(AgentReport.analysis_date.desc()).limit(limit)
            if agent_id:
                stmt = stmt.where(AgentReport.agent_id == agent_id)
            return list(self._session.scalars(stmt).all())
        except Exception:
            return []

    # ── 交易 ────────────────────────────────────────────────

    def get_position(self, code: str) -> Optional[Position]:
        """查询单只持仓（按股票代码）。"""
        try:
            stmt = select(Position).where(Position.code == code)
            return self._session.scalar(stmt)
        except Exception:
            self._set_degraded()
            return None

    def get_all_positions(self) -> List[Position]:
        """获取全部持仓。"""
        try:
            stmt = select(Position).where(Position.quantity > 0)
            return list(self._session.scalars(stmt).all())
        except Exception:
            return []

    def upsert_position(self, code: str, quantity: int, avg_cost: Decimal, current_price: Decimal | None = None) -> Position:
        """更新或插入持仓（按股票代码匹配）。"""
        try:
            stmt = select(Position).where(Position.code == code)
            existing = self._session.scalar(stmt)
            if existing:
                existing.quantity = quantity
                existing.avg_cost = avg_cost
                if current_price is not None:
                    existing.current_price = current_price
                    existing.market_value = Decimal(quantity) * current_price
                existing.updated_at = utcnow()
                return existing
            market_value = Decimal(quantity) * current_price if current_price is not None else None
            pos = Position(code=code, quantity=quantity, avg_cost=avg_cost, current_price=current_price, market_value=market_value)
            self.add(pos)
            return pos
        except Exception:
            self._set_degraded()
            # 返回未持久化的占位对象
            pos = Position(code=code, quantity=quantity, avg_cost=avg_cost, current_price=current_price, market_value=None)
            return pos

    # ── 风控 ────────────────────────────────────────────────

    def log_risk(self, level: str, category: str, message: str, detail: str | None = None) -> RiskLog:
        """记录风控日志。"""
        log = RiskLog(timestamp=utcnow(), level=level, category=category, message=message, detail=detail)
        self.add(log)
        return log

    def log_circuit_breaker(self, from_state: str, to_state: str, reason: str, metrics: str | None = None) -> CircuitBreakerEvent:
        """记录熔断器状态变更。"""
        event = CircuitBreakerEvent(
            timestamp=utcnow(),
            from_state=from_state,
            to_state=to_state,
            trigger_reason=reason,
            metrics=metrics,
        )
        self.add(event)
        return event

    @property
    def is_degraded(self) -> bool:
        """当前是否处于降级模式。"""
        return self._degraded
