"""tushenjing 图神经网络引擎 Prometheus 指标。"""
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

graph_build_duration = Histogram("lingshu_graph_build_duration_seconds", "Graph building duration", buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0], registry=REGISTRY)
graph_nodes_total = Gauge("lingshu_graph_nodes_total", "Total nodes in graph", registry=REGISTRY)
graph_edges_total = Gauge("lingshu_graph_edges_total", "Total edges in graph", ["edge_type"], registry=REGISTRY)
gnn_train_loss = Gauge("lingshu_gnn_train_loss", "Current training loss", registry=REGISTRY)
gnn_val_loss = Gauge("lingshu_gnn_val_loss", "Current validation loss", registry=REGISTRY)
gnn_train_epochs_total = Counter("lingshu_gnn_train_epochs_total", "Total training epochs", registry=REGISTRY)
gnn_early_stopped = Gauge("lingshu_gnn_early_stopped", "Whether training early-stopped", registry=REGISTRY)
gnn_inference_duration = Histogram("lingshu_gnn_inference_duration_seconds", "GNN inference duration", buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0], registry=REGISTRY)
gnn_inference_nodes_total = Counter("lingshu_gnn_inference_nodes_total", "Total nodes processed in inference", registry=REGISTRY)
graph_update_operations_total = Counter("lingshu_graph_update_operations_total", "Total graph update operations", ["operation"], registry=REGISTRY)
