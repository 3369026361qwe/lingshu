/**
 * Hooks 单元测试 — useAPI / useWebSocket / useInterval。
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// ── useAPI ──────────────────────────────────────────────

import { useAPI } from '../hooks/useAPI';

describe('useAPI', () => {
  it('initial state: loading=true, data=null, error=null', () => {
    const fetcher = vi.fn().mockResolvedValue({ picks: [] });
    const { result } = renderHook(() => useAPI(fetcher));
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('sets data and loading=false on success', async () => {
    const fetcher = vi.fn().mockResolvedValue({ picks: [{ code: '000001' }] });
    const { result } = renderHook(() => useAPI(fetcher));
    await waitFor(() => { expect(result.current.loading).toBe(false); });
    expect(result.current.data).toEqual({ picks: [{ code: '000001' }] });
    expect(result.current.error).toBeNull();
  });

  it('sets error on fetch failure', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('Network error'));
    const { result } = renderHook(() => useAPI(fetcher));
    await waitFor(() => { expect(result.current.loading).toBe(false); });
    expect(result.current.error).toBe('Network error');
    expect(result.current.data).toBeNull();
  });

  it('refetch updates data', async () => {
    let call = 0;
    const fetcher = vi.fn().mockImplementation(() => Promise.resolve({ count: ++call }));
    const { result } = renderHook(() => useAPI(fetcher));
    await waitFor(() => { expect(result.current.loading).toBe(false); });
    expect(result.current.data).toEqual({ count: 1 });

    await act(async () => { result.current.refetch(); });
    await waitFor(() => { expect(result.current.data).toEqual({ count: 2 }); });
  });

  it('retries on failure and succeeds', async () => {
    const fetcher = vi.fn()
      .mockRejectedValueOnce(new Error('fail1'))
      .mockResolvedValue({ ok: true });

    const { result } = renderHook(() => useAPI(fetcher, { retry: 2, retryBaseDelay: 1 }));

    await waitFor(() => { expect(result.current.loading).toBe(false); }, { timeout: 5000 });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(result.current.data).toEqual({ ok: true });
  });

  it('gives up after max retries', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('always fail'));
    const { result } = renderHook(() => useAPI(fetcher, { retry: 1, retryBaseDelay: 1 }));

    await waitFor(() => { expect(result.current.loading).toBe(false); }, { timeout: 5000 });
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(result.current.error).toBe('always fail');
  });

  it('calls fetcher with dependency changes', async () => {
    let dep = 1;
    const fetcher = vi.fn().mockResolvedValue({ dep });
    const { rerender } = renderHook(
      ({ d }) => useAPI(fetcher, [d]),
      { initialProps: { d: dep } },
    );
    await waitFor(() => { expect(fetcher).toHaveBeenCalledTimes(1); }, { timeout: 3000 });
    dep = 2;
    rerender({ d: dep });
    await waitFor(() => { expect(fetcher).toHaveBeenCalledTimes(2); }, { timeout: 3000 });
  });
});


// ── useInterval ─────────────────────────────────────────

import useInterval from '../hooks/useInterval';

describe('useInterval', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('calls callback at interval', () => {
    const cb = vi.fn();
    renderHook(() => useInterval(cb, 1000));
    expect(cb).not.toHaveBeenCalled();
    act(() => { vi.advanceTimersByTime(1000); });
    expect(cb).toHaveBeenCalledTimes(1);
    act(() => { vi.advanceTimersByTime(2000); });
    expect(cb).toHaveBeenCalledTimes(3);
  });

  it('does not call when delay is null', () => {
    const cb = vi.fn();
    renderHook(() => useInterval(cb, null));
    act(() => { vi.advanceTimersByTime(5000); });
    expect(cb).not.toHaveBeenCalled();
  });

  it('cleans up on unmount', () => {
    const cb = vi.fn();
    const { unmount } = renderHook(() => useInterval(cb, 1000));
    act(() => { vi.advanceTimersByTime(1000); });
    expect(cb).toHaveBeenCalledTimes(1);
    unmount();
    act(() => { vi.advanceTimersByTime(5000); });
    expect(cb).toHaveBeenCalledTimes(1);
  });

  it('resets when delay changes', () => {
    const cb = vi.fn();
    const { rerender } = renderHook(
      ({ delay }) => useInterval(cb, delay),
      { initialProps: { delay: 1000 as number | null } },
    );
    act(() => { vi.advanceTimersByTime(500); });
    rerender({ delay: 2000 });
    act(() => { vi.advanceTimersByTime(1500); });
    expect(cb).not.toHaveBeenCalled(); // timer reset, 500+1500 < new 2000
    act(() => { vi.advanceTimersByTime(1000); });
    expect(cb).toHaveBeenCalledTimes(1);
  });
});


// ── useWebSocket ────────────────────────────────────────

import useWebSocket from '../hooks/useWebSocket';

describe('useWebSocket', () => {
  let mockWS: any;
  let wsInstances: any[] = [];

  beforeEach(() => {
    vi.useFakeTimers();
    wsInstances = [];
    mockWS = vi.fn(function (this: any, _url: string) {
      this.readyState = 0;
      this.onopen = null;
      this.onmessage = null;
      this.onclose = null;
      this.onerror = null;
      this.close = vi.fn();
      wsInstances.push(this);
      return this;
    });
    (globalThis as any).WebSocket = mockWS;
  });

  afterEach(() => {
    vi.useRealTimers();
    delete (globalThis as any).WebSocket;
  });

  it('initial state: data=null, connected=false', () => {
    const { result } = renderHook(() => useWebSocket('/ws/market'));
    expect(result.current.data).toBeNull();
    expect(result.current.connected).toBe(false);
  });

  it('creates WebSocket with correct URL', () => {
    renderHook(() => useWebSocket('/ws/test'));
    expect(mockWS).toHaveBeenCalled();
    const url = mockWS.mock.calls[0][0];
    expect(url).toContain('/ws/test');
  });

  it('handles incoming JSON message', () => {
    const { result } = renderHook(() => useWebSocket('/ws/test'));
    act(() => {
      wsInstances[0].onmessage?.({ data: JSON.stringify({ type: 'market', data: { index: '5000' } }) } as MessageEvent);
    });
    expect(result.current.data).toEqual({ type: 'market', data: { index: '5000' } });
  });

  it('handles malformed JSON gracefully', () => {
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { result } = renderHook(() => useWebSocket('/ws/test'));
    act(() => {
      wsInstances[0].onmessage?.({ data: 'not json' } as MessageEvent);
    });
    expect(result.current.data).toBeNull();
    expect(consoleWarn).toHaveBeenCalled();
    consoleWarn.mockRestore();
  });

  it('cleans up on unmount', () => {
    const { unmount } = renderHook(() => useWebSocket('/ws/test'));
    const ws = wsInstances[0];
    unmount();
    expect(ws.onclose).toBeNull(); // prevent reconnect
    expect(ws.close).toHaveBeenCalled();
  });
});
