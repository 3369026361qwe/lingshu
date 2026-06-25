import { useState, useEffect, useCallback, useRef } from 'react';

interface UseAPIOptions {
  /** 失败重试次数，默认 0（不重试） */
  retry?: number;
  /** 重试间隔 ms（翻倍递增），默认 1000 */
  retryBaseDelay?: number;
}

/**
 * 通用数据请求 Hook — 支持自动重试。
 *
 * Usage:
 *   const { data, loading, error, refetch } = useAPI(
 *     () => api.get<SelectionResponse>('/api/selection'),
 *     { retry: 2 },
 *   );
 */
export function useAPI<T>(
  fetcher: () => Promise<T>,
  depsOrOptions: unknown[] | UseAPIOptions = [],
) {
  // 兼容旧签名: useAPI(fetcher, deps) 和 useAPI(fetcher, { retry })
  const deps: unknown[] = Array.isArray(depsOrOptions) ? depsOrOptions : [];
  const opts: UseAPIOptions = Array.isArray(depsOrOptions) ? {} : depsOrOptions;
  const { retry = 0, retryBaseDelay = 1000 } = opts;

  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  const retryCountRef = useRef(0);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);

    let lastErr: Error | null = null;
    const maxAttempts = 1 + retry;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const result = await fetcher();
        if (mountedRef.current) {
          setData(result);
          retryCountRef.current = 0;
          setError(null);
        }
        break; // 成功，跳出循环
      } catch (err) {
        lastErr = err as Error;
        if (attempt < maxAttempts - 1) {
          // 指数退避等待
          const delay = retryBaseDelay * Math.pow(2, attempt);
          await new Promise(r => setTimeout(r, delay));
        }
      }
    }

    if (lastErr && mountedRef.current) {
      retryCountRef.current += 1;
      setError(lastErr.message);
    }

    if (mountedRef.current) {
      setLoading(false);
    }
  }, [fetcher, retry, retryBaseDelay, ...deps]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    mountedRef.current = true;
    fetch();
    return () => { mountedRef.current = false; };
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

export default useAPI;
