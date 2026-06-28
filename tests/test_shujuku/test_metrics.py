"""
测试 shujuku.metrics 模块：Prometheus 指标注册与系统信息。
"""

from prometheus_client import REGISTRY


class TestMetricsRegistration:
    """验证所有指标已正确注册到 Prometheus REGISTRY。"""

    def test_db_metrics_registered(self):
        metrics = {m.name for m in REGISTRY.collect()}
        # Prometheus 自动去除 Counter 的 _total 后缀
        expected = {
            "lingshu_db_operations",
            "lingshu_db_operation_latency_seconds",
            "lingshu_db_errors",
            "lingshu_cache_hits",
            "lingshu_cache_misses",
            "lingshu_redis_available",
            "lingshu_db_session_pool_size",
            "lingshu_db_degraded",
            "lingshu_cache_backend_active",
            "lingshu_system",
        }
        missing = expected - metrics
        assert not missing, f"Missing metrics: {missing}"

    def test_system_info_has_version(self):
        from shujuku.metrics import _system_info
        samples = list(_system_info.collect()[0].samples)
        # Info 指标将 key=value 对作为 labels，value 固定为 1.0
        version_labels = [s for s in samples if s.labels.get("version") == "2.0.0"]
        assert len(version_labels) > 0, f"No version label found in samples: {[(s.name, s.labels) for s in samples]}"
        module_labels = [s for s in samples if s.labels.get("module") == "shujuku"]
        assert len(module_labels) > 0

    def test_db_ops_counter_labels(self):
        from shujuku.metrics import db_ops_total
        db_ops_total.labels(operation="insert", table="test").inc()
        # 验证指标确实存在并可以递增
        samples = [
            s for s in db_ops_total.collect()[0].samples
            if s.labels.get("operation") == "insert"
        ]
        assert len(samples) > 0

    def test_degraded_gauges(self):
        from shujuku.metrics import cache_backend_active, db_degraded
        # 验证 gauge 存在且可设置
        db_degraded.set(1)
        cache_backend_active.set(0)
        # 读取回验证
        assert db_degraded._value.get() == 1.0
        assert cache_backend_active._value.get() == 0.0
        # 重置
        db_degraded.set(0)
        cache_backend_active.set(1)
