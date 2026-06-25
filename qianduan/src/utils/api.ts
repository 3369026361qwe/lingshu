/**
 * HTTP 请求封装层 — 基于 fetch，零额外依赖。
 *
 * 功能:
 *   - 自动 JSON 解析 + 错误标准化
 *   - 请求超时 (默认 15s)
 *   - GET 参数自动拼接
 *   - 全类型安全的请求/响应
 */

const BASE_URL = '';           // Vite proxy 处理 /api 前缀，无需设置
const DEFAULT_TIMEOUT = 15_000; // 15 秒

// ── 自定义错误类型 ──────────────────────────────────

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public data?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class TimeoutError extends Error {
  constructor(url: string, timeoutMs: number) {
    super(`请求超时: ${url} (${timeoutMs}ms)`);
    this.name = 'TimeoutError';
  }
}

// ── 核心请求函数 ────────────────────────────────────

async function request<T>(
  method: string,
  path: string,
  options?: {
    body?: unknown;
    params?: Record<string, string | number | undefined>;
    timeout?: number;
    headers?: Record<string, string>;
  },
): Promise<T> {
  const { body, params, timeout = DEFAULT_TIMEOUT, headers: extraHeaders } = options ?? {};

  // 拼接查询参数
  let url = path.startsWith('http') ? path : `${BASE_URL}${path}`;
  if (params) {
    const sp = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== '') sp.append(k, String(v));
    }
    const qs = sp.toString();
    if (qs) url += `?${qs}`;
  }

  // 超时控制
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const res = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...extraHeaders,
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    // 无内容响应 (204 等)
    if (res.status === 204) return undefined as T;

    const json = await res.json();

    if (!res.ok) {
      throw new ApiError(
        res.status,
        json?.error || json?.detail || `HTTP ${res.status}`,
        json,
      );
    }

    return json as T;
  } catch (err) {
    if (err instanceof ApiError || err instanceof TimeoutError) throw err;
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new TimeoutError(url, timeout);
    }
    // 网络错误
    throw new ApiError(0, `网络请求失败: ${(err as Error).message}`);
  } finally {
    clearTimeout(timer);
  }
}

// ── 公开方法 ────────────────────────────────────────

export const api = {
  get<T>(path: string, params?: Record<string, string | number | undefined>) {
    return request<T>('GET', path, { params });
  },

  post<T>(path: string, body?: unknown) {
    return request<T>('POST', path, { body });
  },

  put<T>(path: string, body?: unknown) {
    return request<T>('PUT', path, { body });
  },

  delete<T>(path: string) {
    return request<T>('DELETE', path);
  },

  /** 上传文件或其他非 JSON 请求 */
  raw(
    method: string,
    path: string,
    options?: { body?: BodyInit; headers?: Record<string, string>; timeout?: number },
  ) {
    return request<Response>(method, path, {
      ...options,
      headers: { ...options?.headers, 'Content-Type': '' }, // 覆盖 Content-Type
    });
  },
};

export default api;
