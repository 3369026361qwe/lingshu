"""fengkong 风控模块 Prometheus 指标。"""
from prometheus_client import REGISTRY, Counter, Gauge

breaker_state = Gauge("lingshu_breaker_state", "Circuit breaker state (0=CLOSED,1=OPEN,2=HALF_OPEN)", registry=REGISTRY)
breaker_transitions = Counter("lingshu_breaker_transitions_total", "Breaker state transitions", ["from_state","to_state"], registry=REGISTRY)
rate_limiter_blocks = Counter("lingshu_rate_limiter_blocks_total", "Rate limiter blocked requests", registry=REGISTRY)
position_violations = Counter("lingshu_position_violations_total", "Position limit violations", ["type"], registry=REGISTRY)
var_95_value = Gauge("lingshu_var_95", "Latest VaR 95% value", registry=REGISTRY)
cvar_95_value = Gauge("lingshu_cvar_95", "Latest CVaR 95% value", registry=REGISTRY)
stress_max_loss = Gauge("lingshu_stress_max_loss", "Maximum stress test loss ratio", registry=REGISTRY)
risk_check_total = Counter("lingshu_risk_check_total", "Total risk checks", ["result"], registry=REGISTRY)
