"""依赖注入 — 单例管理。"""
from shujuku.repository import Repository
from shujuku.session import get_session

_repo_singleton = None


def get_repository() -> Repository:
    global _repo_singleton
    if _repo_singleton is None:
        _repo_singleton = Repository(get_session())
    return _repo_singleton
