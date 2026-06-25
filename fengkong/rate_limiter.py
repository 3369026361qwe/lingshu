"""滑动窗口频率限制器。"""
import time as _time
from collections import deque


class RateLimiter:
    """滑动窗口频率限制。"""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def acquire(self) -> bool:
        """尝试获取许可。返回 True 表示允许通过。"""
        now = _time.time()
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) < self.max_requests:
            self._timestamps.append(now)
            return True
        return False

    @property
    def remaining(self) -> int:
        cutoff = _time.time() - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        return self.max_requests - len(self._timestamps)

    @property
    def is_limited(self) -> bool:
        return not self.acquire()
