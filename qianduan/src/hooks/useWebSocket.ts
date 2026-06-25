import { useEffect, useState, useRef, useCallback } from 'react';
import type { WSMessage } from '../types/api';

const MAX_RETRIES = 5;
const BASE_DELAY = 3000; // 起始重连间隔 3s，指数退避

/**
 * WebSocket 连接 Hook — 自动重连 + 类型安全 + 错误日志。
 */
function useWebSocket(path: string) {
  const [data, setData] = useState<WSMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    // 清理旧连接
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onclose = null;
      wsRef.current.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}${path}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      retryRef.current = 0; // 连接成功后重置重试计数
    };

    ws.onmessage = (e) => {
      try {
        const parsed: WSMessage = JSON.parse(e.data);
        setData(parsed);
      } catch (err) {
        console.warn('[WebSocket] 消息解析失败:', err, 'raw:', e.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // 指数退避重连 (上限 5 次)
      if (retryRef.current < MAX_RETRIES) {
        const delay = BASE_DELAY * Math.pow(2, retryRef.current);
        retryRef.current += 1;
        console.info(`[WebSocket] ${path} 断开，${delay}ms 后第 ${retryRef.current} 次重连...`);
        timerRef.current = setTimeout(connect, delay);
      } else {
        console.warn(`[WebSocket] ${path} 重连已达上限 (${MAX_RETRIES})，放弃`);
      }
    };

    ws.onerror = () => {
      // onerror 之后会触发 onclose，由 onclose 处理重连
      console.warn(`[WebSocket] ${path} 连接错误`);
    };
  }, [path]);

  useEffect(() => {
    connect();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // 阻止 cleanup 时触发重连
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { data, connected };
}

export default useWebSocket;
