/**
 * Mock 兜底数据 — 供 Dashboard / Selection / Risk 页面使用。
 *
 * 当后端 API 不可用或返回空数据时，用这些预置数据保证前端渲染正常。
 * 后端上线后可安全删除此文件。
 */

import type { AgentReport, MarketData } from '../types/api';

// ── 权益曲线 ─────────────────────────────────────────

export const FALLBACK_EQUITY = {
  dates: ['05/18', '05/20', '05/22', '05/24', '05/26', '05/28', '05/30',
          '06/02', '06/04', '06/06', '06/08', '06/10', '06/12', '06/14'],
  values: [0, 1.8, 3.2, 2.9, 5.1, 7.3, 6.8, 9.5, 8.2, 10.1, 12.4, 11.0, 14.2, 15.8],
  bench: [0, 0.9, 1.5, 2.1, 1.8, 2.8, 3.2, 4.1, 3.8, 5.0, 5.8, 5.5, 6.8, 7.5],
};

// ── 选股 ─────────────────────────────────────────────

export const MOCK_PICKS = [
  { code: '300750', score: 94.2, rank: 1 },
  { code: '002594', score: 91.8, rank: 2 },
  { code: '600519', score: 89.3, rank: 3 },
  { code: '300274', score: 87.6, rank: 4 },
  { code: '000858', score: 85.2, rank: 5 },
];

// ── 智能体 ───────────────────────────────────────────

export const MOCK_AGENTS: AgentReport[] = [
  { agent_id: 'macro', timestamp: '', signal: 'bullish', confidence: '0.78',
    reasoning: 'PMI连续3月扩张，建议超配制造业' },
  { agent_id: 'sector', timestamp: '', signal: 'bullish', confidence: '0.82',
    reasoning: '新能源板块资金持续流入，光伏和锂电是当前最强赛道' },
  { agent_id: 'risk', timestamp: '', signal: 'bearish', confidence: '0.85',
    reasoning: '北向资金连续3日净流出，建议总仓位降至80%' },
];

// ── 因子权重 ─────────────────────────────────────────

export const FALLBACK_WEIGHTS = [
  { name: 'GNN', weight: 0.22, source: 'mock' as const },
  { name: 'Agent', weight: 0.25, source: 'mock' as const },
  { name: 'ROE', weight: 0.18, source: 'mock' as const },
  { name: 'PE', weight: 0.15, source: 'mock' as const },
  { name: '动量', weight: 0.12, source: 'mock' as const },
  { name: '情绪', weight: 0.08, source: 'mock' as const },
];

// ── 市场概览（三指数） ───────────────────────────────

export const FALLBACK_MARKET: MarketData = {
  latest_date: '',
  stock_count: 0,
  avg_change_pct: 0.82,
  csi300: { index: '沪深300', value: '3,856.21', change: '+0.82%', up: true },
  csi500: { index: '中证500', value: '5,821.35', change: '+1.15%', up: true },
  chinext: { index: '创业板指', value: '1,892.45', change: '+2.03%', up: true },
  updated_at: '',
};
