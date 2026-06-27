/**
 * 选股列表全局 store — 跨页面共享 Top-N 选股结果。
 */
import { create } from 'zustand';

interface Pick {
  code: string;
  score: number;
  rank: number;
}

interface SelectionState {
  date: string;
  picks: Pick[];
  count: number;
  setSelection: (data: { date?: string; picks?: Pick[]; count?: number }) => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  date: '',
  picks: [],
  count: 0,
  setSelection: (data) =>
    set({
      date: data.date ?? '',
      picks: data.picks ?? [],
      count: data.count ?? 0,
    }),
}));
