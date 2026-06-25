import React from 'react';

interface ScoreBarProps {
  label: string;
  value: number;          // 0–1
  color?: string;
  maxWidth?: number;       // 参照最大值
}

/** 因子权重 / 得分进度条 */
const ScoreBar: React.FC<ScoreBarProps> = ({ label, value, color = '#4198FF', maxWidth = 1 }) => (
  <div style={{ marginBottom: 8 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', color: '#E8EAED', fontSize: 13, marginBottom: 2 }}>
      <span>{label}</span>
      <span style={{ fontWeight: 600, color }}>{(value * 100).toFixed(0)}%</span>
    </div>
    <div style={{ height: 6, background: '#21262D', borderRadius: 3, overflow: 'hidden' }}>
      <div style={{
        height: '100%',
        width: `${Math.min((value / maxWidth) * 100, 100)}%`,
        background: color,
        borderRadius: 3,
        transition: 'width 0.4s ease',
      }} />
    </div>
  </div>
);

export default ScoreBar;
