import React from 'react';
import { Tag } from 'antd';
import { riskColor } from '../utils/format';

interface RiskBadgeProps {
  level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  showLabel?: boolean;
}

const LEVEL_LABEL: Record<string, string> = {
  LOW: '低风险',
  MEDIUM: '中风险',
  HIGH: '高风险',
  CRITICAL: '极高风险',
};

const RiskBadge: React.FC<RiskBadgeProps> = ({ level, showLabel = true }) => (
  <Tag color={riskColor(level)} style={{ fontWeight: 600, fontSize: 14 }}>
    {showLabel ? LEVEL_LABEL[level] ?? level : level}
  </Tag>
);

export default RiskBadge;
