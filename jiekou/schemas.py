"""Pydantic V2 请求/响应模型。"""
from datetime import datetime

from pydantic import BaseModel, Field


class StockInfoResponse(BaseModel):
    code: str
    name: str
    exchange: str = "SZ"
    is_active: bool = True


class DailyBarResponse(BaseModel):
    code: str
    trade_date: str
    open: str
    high: str
    low: str
    close: str
    volume: str
    amount: str
    turnover_rate: str | None = None


class FactorValueResponse(BaseModel):
    code: str
    factor_name: str
    category: str
    raw_value: str
    z_score: str | None = None
    percentile: str | None = None


class AgentReportResponse(BaseModel):
    agent_id: str
    timestamp: datetime
    signal: str
    confidence: str
    reasoning: str
    risk_flags: list[str] = []
    target_stocks: list[str] = []


class PortfolioResponse(BaseModel):
    code: str
    quantity: int
    weight: str
    market_value: str
    avg_cost: str
    current_price: str | None = None


class SelectionRequest(BaseModel):
    stock_list: list[str] = []
    top_n: int = Field(default=20, ge=1, le=100)
    exclude: list[str] = []


class SelectionResponse(BaseModel):
    picks: list[dict]
    composite_scores: dict[str, str]
    timestamp: datetime


class RiskStatusResponse(BaseModel):
    risk_level: str
    risk_score: int
    blocked: bool
    breaker_state: str
    position_violations: list[str] = []
    var_95: str | None = None
    advice: str


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    initial_capital: str = "1000000"
    strategy_params: dict = {}
