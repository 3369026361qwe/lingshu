"""jiekou API 层 Prometheus 指标。"""
from prometheus_client import Counter, Histogram, REGISTRY
http_requests_total = Counter("lingshu_http_requests_total", "HTTP requests", ["method","endpoint","status"], registry=REGISTRY)
http_request_duration = Histogram("lingshu_http_request_duration_seconds", "HTTP request duration", ["method","endpoint"], buckets=[0.001,0.01,0.05,0.1,0.5,1,5], registry=REGISTRY)
ws_connections = Counter("lingshu_ws_connections_total", "WebSocket connections", ["channel"], registry=REGISTRY)
