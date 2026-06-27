/**
 * 市场概览全局状态 — WebSocket 推送的市场数据跨页面共享。
 *
 * DashboardPage 接收 /ws/market 数据后写入此 store，
 * 其他页面可直接读取，避免重复 WebSocket 连接。
 */
import { create } from 'zustand';

interface MarketState {
  index: string;
  change: string;
  latestDate: string;
  stockCount: number;
  avgChangePct: number;
  updatedAt: string;
  setMarket: (data: Record<string, unknown>) => void;
}

export const useMarketStore = create<MarketState>((set) => ({
  index: '3,856.21',
  change: '+0.82%',
  latestDate: '',
  stockCount: 0,
  avgChangePct: 0,
  updatedAt: '',
  setMarket: (data) =>
    set({
      index: (data.index as string) ?? '3,856.21',
      change: (data.change as string) ?? '+0.82%',
      latestDate: (data.latest_date as string) ?? '',
      stockCount: (data.stock_count as number) ?? 0,
      avgChangePct: (data.avg_change_pct as number) ?? 0,
      updatedAt: (data.updated_at as string) ?? '',
    }),
}));
