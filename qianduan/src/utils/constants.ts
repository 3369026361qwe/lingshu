/** 行业色板等常量。 */

export const INDUSTRY_COLORS: Record<string, string> = {
  '电力设备': '#4198FF',
  '汽车': '#00D4AA',
  '食品饮料': '#FF8C42',
  '医药生物': '#FF475C',
  '电子': '#D4AF37',
  '计算机': '#2ECC71',
  '机械设备': '#9B59B6',
  '基础化工': '#1ABC9C',
  '银行': '#E74C3C',
  '非银金融': '#3498DB',
  '房地产': '#95A5A6',
  '国防军工': '#E67E22',
};

export function industryColor(name: string): string {
  return INDUSTRY_COLORS[name] || '#4198FF';
}

/** 页面刷新间隔 (毫秒) */
export const POLL_INTERVAL = {
  market: 10_000,
  risk: 15_000,
  agents: 30_000,
  backtest: 3_000,
} as const;
