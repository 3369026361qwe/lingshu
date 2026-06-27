/**
 * Zustand Stores 单元测试 — marketStore / riskStore / selectionStore。
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';

// ── marketStore ─────────────────────────────────────────

import { useMarketStore } from '../stores/marketStore';

describe('marketStore', () => {
  beforeEach(() => {
    act(() => { useMarketStore.setState(useMarketStore.getInitialState()); });
  });

  it('has default values', () => {
    const state = useMarketStore.getState();
    expect(state.index).toBe('3,856.21');
    expect(state.change).toBe('+0.82%');
    expect(state.stockCount).toBe(0);
    expect(state.latestDate).toBe('');
  });

  it('setMarket updates all fields', () => {
    act(() => {
      useMarketStore.getState().setMarket({
        index: '4,000.00',
        change: '+1.50%',
        stock_count: 5000,
        latest_date: '2026-06-28',
        avg_change_pct: 1.2,
        updated_at: '2026-06-28T15:00:00Z',
      });
    });
    const state = useMarketStore.getState();
    expect(state.index).toBe('4,000.00');
    expect(state.change).toBe('+1.50%');
    expect(state.stockCount).toBe(5000);
    expect(state.latestDate).toBe('2026-06-28');
  });

  it('setMarket falls back to defaults for missing fields', () => {
    act(() => {
      useMarketStore.getState().setMarket({});
    });
    const state = useMarketStore.getState();
    expect(state.index).toBe('3,856.21');
    expect(state.stockCount).toBe(0);
  });
});


// ── riskStore ───────────────────────────────────────────

import { useRiskStore } from '../stores/riskStore';

describe('riskStore', () => {
  beforeEach(() => {
    act(() => { useRiskStore.setState(useRiskStore.getInitialState()); });
  });

  it('has default values', () => {
    const state = useRiskStore.getState();
    expect(state.riskLevel).toBe('LOW');
    expect(state.riskScore).toBe(0);
    expect(state.blocked).toBe(false);
    expect(state.breakerState).toBe('CLOSED');
    expect(state.var95).toBe('1.2%');
  });

  it('setRisk updates risk state', () => {
    act(() => {
      useRiskStore.getState().setRisk({
        risk_level: 'HIGH',
        risk_score: 0.75,
        blocked: true,
        breaker_state: 'HALF_OPEN',
        var_95: '2.5%',
      });
    });
    const state = useRiskStore.getState();
    expect(state.riskLevel).toBe('HIGH');
    expect(state.riskScore).toBe(0.75);
    expect(state.blocked).toBe(true);
    expect(state.breakerState).toBe('HALF_OPEN');
    expect(state.var95).toBe('2.5%');
  });

  it('setRisk handles partial data', () => {
    act(() => { useRiskStore.getState().setRisk({ risk_level: 'CRITICAL' }); });
    const state = useRiskStore.getState();
    expect(state.riskLevel).toBe('CRITICAL');
    expect(state.blocked).toBe(false); // unchanged
  });
});


// ── selectionStore ──────────────────────────────────────

import { useSelectionStore } from '../stores/selectionStore';

describe('selectionStore', () => {
  beforeEach(() => {
    act(() => { useSelectionStore.setState(useSelectionStore.getInitialState()); });
  });

  it('has default values', () => {
    const state = useSelectionStore.getState();
    expect(state.picks).toEqual([]);
    expect(state.date).toBe('');
    expect(state.count).toBe(0);
  });

  it('setSelection updates selection data', () => {
    act(() => {
      useSelectionStore.getState().setSelection({
        picks: [{ code: '000001', score: 95, rank: 1 }, { code: '000002', score: 88, rank: 2 }],
        date: '2026-06-28',
        count: 2,
      });
    });
    const state = useSelectionStore.getState();
    expect(state.picks).toHaveLength(2);
    expect(state.date).toBe('2026-06-28');
    expect(state.count).toBe(2);
  });

  it('setSelection handles undefined picks gracefully', () => {
    act(() => { useSelectionStore.getState().setSelection({ date: '2026-06-28', count: 0 }); });
    const state = useSelectionStore.getState();
    expect(state.picks).toEqual([]);
    expect(state.date).toBe('2026-06-28');
  });
});
