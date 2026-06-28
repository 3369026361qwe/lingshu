"""zhixing 交易执行 Prometheus 指标。"""
from prometheus_client import REGISTRY, Counter

orders_created = Counter("lingshu_orders_created_total", "Orders created", ["direction"], registry=REGISTRY)
orders_filled = Counter("lingshu_orders_filled_total", "Orders filled", ["direction"], registry=REGISTRY)
orders_rejected = Counter("lingshu_orders_rejected_total", "Orders rejected", registry=REGISTRY)
broker_trades_total = Counter("lingshu_broker_trades_total", "Broker trades executed", registry=REGISTRY)
broker_commission_total = Counter("lingshu_broker_commission_total", "Total commission paid", registry=REGISTRY)
executor_batch_total = Counter("lingshu_executor_batch_total", "Batch executions", ["status"], registry=REGISTRY)
recorder_trades_total = Counter("lingshu_recorder_trades_total", "Trades recorded", registry=REGISTRY)
