import { useEffect, useRef } from 'react';

/**
 * 通用定时器 Hook — 轮询、进度检查、定时刷新等场景。
 *
 * Usage:
 *   useInterval(() => refetch(), 10_000);  // 每 10s 调用一次
 *   useInterval(callback, null);            // 暂停
 */
export function useInterval(callback: () => void, delayMs: number | null) {
  const savedCallback = useRef(callback);

  // 始终指向最新的 callback，避免因为 callback 变化而重置定时器
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (delayMs === null) return;

    const id = setInterval(() => savedCallback.current(), delayMs);
    return () => clearInterval(id);
  }, [delayMs]);
}

export default useInterval;
