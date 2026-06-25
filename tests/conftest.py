"""
测试全局配置 — 强制使用独立测试数据库，永不触及生产数据。

关键: 必须在任何 shujuku 模块导入前设置 DATABASE_URL 环境变量。
"""
import os
import sys
from pathlib import Path


def pytest_configure(config):
    """pytest 启动时最先执行 — 切换数据库到测试专用文件。"""
    test_db = str(Path(__file__).resolve().parent.parent / "data" / "test_lingshu.db")

    # 强制覆盖: 测试永远只碰 test_lingshu.db
    os.environ["DATABASE_URL"] = f"sqlite:///{test_db}"
    os.environ["LINGSHU_ENV"] = "dev"  # 允许 drop_all (但仅限 test 数据库)

    # 清除已缓存的 SQLAlchemy engine (如果已导入)
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("shujuku"):
            del sys.modules[mod_name]

    # 确保 test data 目录存在
    os.makedirs(os.path.dirname(test_db), exist_ok=True)


def pytest_unconfigure(config):
    """测试结束清理 — 删除测试数据库（避免残留影响下次测试）。"""
    test_db = str(Path(__file__).resolve().parent.parent / "data" / "test_lingshu.db")
    try:
        os.remove(test_db)
        for suffix in ("-wal", "-shm"):
            p = test_db + suffix
            if os.path.exists(p):
                os.remove(p)
    except FileNotFoundError:
        pass
