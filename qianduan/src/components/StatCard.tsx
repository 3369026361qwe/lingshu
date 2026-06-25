import React from 'react';
import { Card } from 'antd';

interface StatCardProps {
  title: string;
  value: React.ReactNode;
  prefix?: React.ReactNode;
  suffix?: React.ReactNode;
  footer?: React.ReactNode;
  accent?: string;       // 文字颜色
  borderColor?: string;  // 左边框强调色
  loading?: boolean;
}

const StatCard: React.FC<StatCardProps> = ({
  title, value, prefix, suffix, footer, accent = '#E8EAED', borderColor, loading,
}) => (
  <Card
    size="small"
    loading={loading}
    style={{
      background: '#161B22',
      border: borderColor ? `1px solid ${borderColor}` : '1px solid #21262D',
    }}
  >
    <div style={{ color: '#9AA0A6', fontSize: 13, marginBottom: 8 }}>{title}</div>
    <div style={{ fontSize: 20, fontWeight: 600, color: accent }}>
      {prefix}
      {value}
      {suffix}
    </div>
    {footer && <div style={{ color: '#9AA0A6', fontSize: 12, marginTop: 8 }}>{footer}</div>}
  </Card>
);

export default StatCard;
