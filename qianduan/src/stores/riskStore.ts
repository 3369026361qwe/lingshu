/**
 * 风控状态全局 store — 跨页面共享风险等级、断路器状态等。
 */
import { create } from 'zustand';

export interface RiskState {
  riskLevel: string;
  breakerState: string;
  var95: string;
  riskScore: number;
  blocked: boolean;
  advice: string;
  setRisk: (data: Record<string, unknown>) => void;
}

export const useRiskStore = create<RiskState>((set) => ({
  riskLevel: 'LOW',
  breakerState: 'CLOSED',
  var95: '1.2%',
  riskScore: 0,
  blocked: false,
  advice: '',
  setRisk: (data) =>
    set({
      riskLevel: (data.risk_level as string) ?? 'LOW',
      breakerState: (data.breaker_state as string) ?? 'CLOSED',
      var95: (data.var_95 as string) ?? '1.2%',
      riskScore: (data.risk_score as number) ?? 0,
      blocked: (data.blocked as boolean) ?? false,
      advice: (data.advice as string) ?? '',
    }),
}));
