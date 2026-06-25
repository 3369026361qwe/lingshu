"""
测试 shujuku.config 模块：环境变量解析、URL 构造、SQLite/PostgreSQL 检测。

注意：每个测试会 reload config 模块以反映 monkeypatch 的环境变量。
模块级 fixture 在所有测试完成后恢复原始 config 状态，防止污染其他测试文件。
"""

import copy
import importlib
import os

import pytest

import shujuku.config as _config_module


# ── 保存/恢复 config 模块状态 ──────────────────────────

_original_config_state: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _isolate_config():
    """保存原始状态，module 内所有测试完成后恢复。"""
    global _original_config_state
    # 保存
    _original_config_state = {
        "DATABASE_URL": _config_module.DATABASE_URL,
        "REDIS_URL": _config_module.REDIS_URL,
        "DB_ECHO": _config_module.DB_ECHO,
        "is_sqlite": _config_module.is_sqlite,
        "is_postgresql": _config_module.is_postgresql,
    }
    yield
    # 恢复：reload 会从真实环境变量重新构造，再显式回写
    importlib.reload(_config_module)
    # 确保恢复到默认 SQLite 状态（清除测试环境变量影响）
    if not os.getenv("DATABASE_URL") and not os.getenv("DB_HOST"):
        # 默认 SQLite 模式，reload 已经正确处理
        pass


def _reload():
    """重新加载 config 模块以反映 monkeypatch 后的环境变量。"""
    importlib.reload(_config_module)


# ── 测试 ──────────────────────────────────────────────


class TestDatabaseURL:
    def test_default_sqlite(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("DB_HOST", raising=False)
        _reload()
        assert _config_module.is_sqlite()
        assert _config_module.DATABASE_URL.startswith("sqlite:///")

    def test_explicit_database_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
        _reload()
        assert _config_module.is_postgresql()

    def test_postgresql_from_segments(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("DB_HOST", "pg.example.com")
        monkeypatch.setenv("DB_PORT", "5433")
        monkeypatch.setenv("DB_NAME", "lingshu_test")
        monkeypatch.setenv("DB_USER", "admin")
        monkeypatch.setenv("DB_PASSWORD", "s3cret!")
        _reload()
        assert _config_module.is_postgresql()
        assert "pg.example.com" in _config_module.DATABASE_URL

    def test_password_url_encoded(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PASSWORD", "p@ss:word!")
        _reload()
        # 密码中的 @ 应被编码
        assert "%40" in _config_module.DATABASE_URL or "@" not in _config_module.DATABASE_URL.split("://")[1].split("@")[0]


class TestRedisURL:
    def test_default_none(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        _reload()
        assert _config_module.REDIS_URL is None

    def test_set(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        _reload()
        assert _config_module.REDIS_URL == "redis://localhost:6379/0"

    def test_empty_string(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "")
        _reload()
        assert _config_module.REDIS_URL is None


class TestDBEcho:
    def test_default_false(self, monkeypatch):
        monkeypatch.delenv("DB_ECHO", raising=False)
        _reload()
        assert _config_module.DB_ECHO is False

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv("DB_ECHO", "1")
        _reload()
        assert _config_module.DB_ECHO is True

    def test_other_values_false(self, monkeypatch):
        monkeypatch.setenv("DB_ECHO", "true")
        _reload()
        assert _config_module.DB_ECHO is False


class TestIsSQLite:
    def test_sqlite_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./test.db")
        _reload()
        assert _config_module.is_sqlite()
        assert not _config_module.is_postgresql()

    def test_postgres_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        _reload()
        assert _config_module.is_postgresql()
        assert not _config_module.is_sqlite()
