/**
 * API 响应类型定义 — 与后端 jiekou/schemas.py 对齐。
 */

// ── 股票 ───────────────────────────────────────

export interface StockBasic {
  code: string;
  name: string;
  exchange: 'SZ' | 'SH' | 'BJ';
}

export interface DailyBar {
  trade_date: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

// ── 选股 ───────────────────────────────────────

export interface StockPick {
  code: string;
  score: number;
  rank: number;
}

export interface SelectionResponse {
  date: string;
  picks: StockPick[];
  count: number;
  timestamp: string;
}

// ── 智能体报告 ─────────────────────────────────

export interface AgentReport {
  agent_id: string;       // macro | sector | stock | sentiment | risk
  timestamp: string;
  signal: string;         // "bullish" | "bearish" | "neutral"
  confidence: string;     // "0.85"
  reasoning: string;      // 最多 500 字符
}

export type AgentId = 'macro' | 'sector' | 'stock' | 'sentiment' | 'risk';

export const AGENT_LABELS: Record<AgentId, string> = {
  macro: '宏观分析',
  sector: '赛道分析',
  stock: '个股分析',
  sentiment: '舆情分析',
  risk: '风险监控',
};

export const AGENT_COLORS: Record<AgentId, string> = {
  macro: '#4198FF',
  sector: '#2ECC71',
  stock: '#D4AF37',
  sentiment: '#FF8C42',
  risk: '#FF475C',
};

// ── 持仓 ───────────────────────────────────────

export interface Position {
  code: string;
  quantity: number;
  avg_cost: string;
  market_value: string | null;
  weight: string | null;
}

// ── 风控 ───────────────────────────────────────

export interface RiskStatus {
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  risk_score: number;
  blocked: boolean;
  breaker_state: 'CLOSED' | 'HALF_OPEN' | 'OPEN';
  var_95: string | null;
  advice: string;
}

// ── 回测 ───────────────────────────────────────

export interface BacktestConfig {
  start_date: string;
  end_date: string;
  initial_capital: string;
  strategy_params?: Record<string, unknown>;
}

export interface BacktestResult {
  experiment_id: string;
  metrics: Record<string, number>;
  elapsed_seconds: number;
}

export interface EquityPoint {
  date: string;
  value: number;
}

export interface BacktestSummary {
  start_date: string;
  end_date: string;
  total_days: number;
  final_value: number;
  total_return: number;
  max_drawdown: number;
  sharpe: number;
  snapshot_count: number;
  equity_curve: EquityPoint[];
}

export interface EquityResponse {
  data: EquityPoint[];
}

export interface FactorWeight {
  name: string;
  weight: number;
  source: 'kalman' | 'synthetic' | 'mock';
}

export interface FactorWeightsResponse {
  weights: FactorWeight[];
  source: 'live' | 'mock';
}

// ── WebSocket ───────────────────────────────────

export interface WSMessage {
  type: 'market' | 'agent' | 'risk';
  data?: unknown;
  agent_id?: string;
  reasoning?: string;
  confidence?: string;
  risk_level?: string;
  breaker?: string;
  timestamp: string;
}

// ── 市场指数 ───────────────────────────────────

export interface MarketIndex {
  index: string;    // 指数名称
  value: string;    // 点位
  change: string;   // 涨跌幅 "+1.23%"
  up?: boolean;
}

export interface MarketData {
  latest_date: string;
  stock_count: number;
  avg_change_pct: number;
  csi300: MarketIndex;
  csi500: MarketIndex;
  chinext: MarketIndex;
  updated_at: string;
}

// ── 交易 ───────────────────────────────────────

export interface TradeItem {
  code: string;
  action: 'BUY' | 'SELL';
  weight: number;
  delta: number;
  amount: number;
  reason: string;
}

export interface TradePipeline {
  date: string;
  timestamp: string;
  capital: number;
  stocks: { code: string; score: number; rank: number }[];
  portfolio: { code: string; weight: number }[];
  risk: { level: string; score: number; blocked: boolean; advice: string };
  trades: { buys: TradeItem[]; sells: TradeItem[]; buy_count: number; sell_count: number; net_cash_flow: number };
}

export interface TradeOrder {
  code: string;
  direction: string;
  amount: number;
  price: number;
  status: string;
  time: string;
}

// ── GNN 图 ──────────────────────────────────────

export interface GNNNode {
  id: string;
  name: string;
  score: number;
  symbolSize: number;
  itemStyle: { color: string };
}

export interface GNNEdge {
  source: string;
  target: string;
  weight: number;
}

export interface GNNGraphData {
  date: string;
  nodes: GNNNode[];
  edges: GNNEdge[];
  node_count: number;
  edge_count: number;
  error?: string;
}
