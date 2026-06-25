/**
 * 数字/金额/日期格式化工具。
 */

/** 金额格式化 (保留 2 位小数 + 千分位) */
export function formatMoney(v: number | string, decimals = 2): string {
  const n = typeof v === 'string' ? parseFloat(v) : v;
  if (isNaN(n)) return '—';
  return n.toLocaleString('zh-CN', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/** 百分比格式化 */
export function formatPct(v: number, decimals = 2): string {
  if (isNaN(v)) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${(v * 100).toFixed(decimals)}%`;
}

/** 大数字精简 (万/亿)，保留正负号 */
export function formatLargeNum(v: number): string {
  if (isNaN(v)) return '—';
  const sign = v < 0 ? '-' : '';
  const abs = Math.abs(v);
  if (abs >= 1e8) return `${sign}${(abs / 1e8).toFixed(2)}亿`;
  if (abs >= 1e4) return `${sign}${(abs / 1e4).toFixed(2)}万`;
  return `${sign}${abs.toFixed(2)}`;
}

/** 日期格式化: "2026-06-15" → "06-15" */
export function formatDateShort(d: string): string {
  if (!d) return '—';
  return d.slice(5);
}

/** 信号标签 → 颜色 */
export function signalColor(signal: string): string {
  switch (signal) {
    case 'bullish': return '#2ECC71';
    case 'bearish': return '#FF475C';
    default: return '#9AA0A6';
  }
}

/** 信号标签 → 中文 */
export function signalLabel(signal: string): string {
  switch (signal) {
    case 'bullish': return '看多';
    case 'bearish': return '看空';
    default: return '中性';
  }
}

/** 风险等级 → 颜色 */
export function riskColor(level: string): string {
  switch (level) {
    case 'LOW': return '#2ECC71';
    case 'MEDIUM': return '#FF8C42';
    case 'HIGH': return '#FF475C';
    case 'CRITICAL': return '#FF0000';
    default: return '#6B7280';
  }
}
