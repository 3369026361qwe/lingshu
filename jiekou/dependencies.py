"""依赖注入 — 单例管理。"""
from shujuku.session import get_session
from shujuku.repository import Repository

_repo_singleton = None


def get_repository() -> Repository:
    global _repo_singleton
    if _repo_singleton is None:
        _repo_singleton = Repository(get_session())
    return _repo_singleton
