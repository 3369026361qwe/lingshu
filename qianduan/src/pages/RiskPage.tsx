import React, { useMemo } from 'react';
import { Row, Col, Card, Table, Spin, Empty, Progress, Tag, Descriptions } from 'antd';
import { AlertOutlined, ReloadOutlined, SafetyCertificateOutlined, ThunderboltOutlined, WalletOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import api from '../utils/api';
import { riskColor, formatMoney, formatPct } from '../utils/format';
import { POLL_INTERVAL } from '../utils/constants';
import useAPI from '../hooks/useAPI';
import useInterval from '../hooks/useInterval';
import useWebSocket from '../hooks/useWebSocket';
import StatCard from '../components/StatCard';
import RiskBadge from '../components/RiskBadge';
import type { RiskStatus, Position } from '../types/api';

const MOCK_POSITIONS: Position[] = [
  { code: '300750', quantity: 500, avg_cost: '185.50', market_value: '102300.00', weight: '0.082' },
  { code: '002594', quantity: 400, avg_cost: '228.80', market_value: '98752.00', weight: '0.075' },
  { code: '600519', quantity: 50, avg_cost: '1685.00', market_value: '89450.00', weight: '0.071' },
  { code: '300274', quantity: 800, avg_cost: '96.30', market_value: '82080.00', weight: '0.068' },
  { code: '000858', quantity: 400, avg_cost: '152.40', market_value: '65240.00', weight: '0.062' },
];

const RiskPage: React.FC = () => {
  const { data: risk, loading, refetch } = useAPI<RiskStatus>(() => api.get('/api/risk/status'));
  const { data: posData } = useAPI<Position[]>(() => api.get('/api/portfolio'));
  const { data: wsRisk } = useWebSocket('/ws/risk');

  useInterval(() => refetch(), POLL_INTERVAL.risk);

  const positions = posData?.length ? posData : MOCK_POSITIONS;

  // VaR 仪表盘
  const varGaugeOption = useMemo(() => ({
    series: [{
      type: 'gauge',
      startAngle: 210, endAngle: -30,
      center: ['50%', '55%'],
      radius: '90%',
      min: 0, max: 5,
      axisLine: { lineStyle: { width: 20, color: [[0.3, '#2ECC71'], [0.7, '#FF8C42'], [1, '#FF475C']] } },
      pointer: { length: '70%', width: 6, itemStyle: { color: 'auto' } },
      axisTick: { distance: -20, length: 6, lineStyle: { color: '#9AA0A6' } },
      splitLine: { distance: -26, length: 14, lineStyle: { color: '#9AA0A6' } },
      axisLabel: { color: '#9AA0A6', distance: 30, fontSize: 10 },
      detail: { valueAnimation: true, formatter: '{value}%', color: '#E8EAED', fontSize: 18, offsetCenter: [0, '70%'] },
      data: [{ value: parseFloat(risk?.var_95 ?? '1.2'), name: 'VaR (95%)' }],
    }],
  }), [risk]);

  // 仓位饼图
  const positionPie = useMemo(() => ({
    tooltip: { trigger: 'item' },
    series: [{
      type: 'pie',
      radius: ['45%', '75%'],
      center: ['50%', '55%'],
      itemStyle: { borderRadius: 4, borderColor: '#0D1117', borderWidth: 2 },
      label: { color: '#9AA0A6', fontSize: 11 },
      data: positions.map((p, i) => ({
        value: parseFloat(p.weight ?? '0') * 100,
        name: p.code,
        itemStyle: { color: ['#D4AF37', '#4198FF', '#2ECC71', '#FF8C42', '#FF475C', '#00D4AA'][i % 6] },
      })),
    }],
  }), [positions]);

  const breakerState = risk?.breaker_state ?? 'CLOSED';
  const riskScore = risk?.risk_score ? (risk.risk_score * 100).toFixed(0) : '15';

  return (
    <Spin spinning={loading}>
      <Row gutter={[16, 16]}>
        {/* ── 风险概览 ── */}
        <Col span={6}>
          <StatCard
            title="⚠ 风险等级"
            value={<RiskBadge level={(risk?.risk_level ?? 'LOW') as RiskStatus['risk_level']} />}
            accent={riskColor(risk?.risk_level ?? 'LOW')}
            borderColor={riskColor(risk?.risk_level ?? 'LOW')}
            footer={`${risk?.blocked ? '🔴 已被风控拦截' : '🟢 风控正常'}`}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title="📉 VaR (95% 1-Day)"
            value={`${risk?.var_95 ?? '1.2'}%`}
            accent="#FF8C42"
            footer="历史模拟法"
          />
        </Col>
        <Col span={6}>
          <StatCard
            title="⚡ 熔断器状态"
            value={breakerState}
            accent={breakerState === 'OPEN' ? '#FF475C' : breakerState === 'HALF_OPEN' ? '#FF8C42' : '#2ECC71'}
            prefix={breakerState === 'OPEN' ? <ThunderboltOutlined /> : <SafetyCertificateOutlined />}
          />
        </Col>
        <Col span={6}>
          <StatCard
            title="📋 风控建议"
            value={risk?.advice ?? '正常'}
            accent="#4198FF"
          />
        </Col>

        {/* ── VaR 仪表盘 + 仓位饼图 ── */}
        <Col span={8}>
          <Card style={{ background: '#161B22', border: '1px solid #21262D' }}>
            <ReactECharts option={varGaugeOption} style={{ height: 280 }} />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="💰 仓位分布" style={{ background: '#161B22', border: '1px solid #21262D' }}>
            {positions.length ? (
              <ReactECharts option={positionPie} style={{ height: 280 }} />
            ) : (
              <Empty description="暂无持仓数据" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            )}
          </Card>
        </Col>
        <Col span={8}>
          <Card title={<span><AlertOutlined /> 风险评分</span>} style={{ background: '#161B22', border: '1px solid #21262D', height: '100%' }}>
            <div style={{ textAlign: 'center', padding: '40px 0' }}>
              <Progress
                type="dashboard"
                percent={parseFloat(riskScore)}
                strokeColor={{ '0%': '#2ECC71', '70%': '#FF8C42', '100%': '#FF475C' } as any}
                format={() => <span style={{ color: riskColor(risk?.risk_level ?? 'LOW'), fontSize: 24, fontWeight: 700 }}>{riskScore}%</span>}
                size={180}
              />
              <div style={{ color: '#9AA0A6', marginTop: 16, fontSize: 13 }}>
                {parseFloat(riskScore) < 30 ? '✅ 风险可控' : parseFloat(riskScore) < 60 ? '⚠ 关注风险' : '🔴 需要减仓'}
              </div>
            </div>
          </Card>
        </Col>

        {/* ── 持仓表格 ── */}
        <Col span={24}>
          <Card
            title={<span><WalletOutlined /> 当前持仓</span>}
            extra={<ReloadOutlined onClick={() => refetch()} style={{ cursor: 'pointer', color: '#9AA0A6' }} />}
            style={{ background: '#161B22', border: '1px solid #21262D' }}
          >
            <Table
              dataSource={positions}
              rowKey="code"
              size="small"
              pagination={false}
              columns={[
                { title: '代码', dataIndex: 'code', key: 'code', render: (v: string) => <span style={{ color: '#D4AF37', fontFamily: 'monospace', fontWeight: 600 }}>{v}</span>, width: 96 },
                { title: '数量', dataIndex: 'quantity', key: 'quantity', align: 'right' as const },
                { title: '成本价', dataIndex: 'avg_cost', key: 'avg_cost', align: 'right' as const, render: (v: string) => formatMoney(v) },
                { title: '市值', dataIndex: 'market_value', key: 'market_value', align: 'right' as const, render: (v: string | null) => v ? formatMoney(v) : '—' },
                { title: '权重', dataIndex: 'weight', key: 'weight', align: 'right' as const, render: (v: string | null) => {
                  const pct = parseFloat(v ?? '0');
                  return <span style={{ color: pct > 0.08 ? '#FF8C42' : '#2ECC71', fontWeight: 600 }}>{formatPct(pct)}</span>;
                }},
              ]}
              style={{ background: 'transparent' }}
            />
          </Card>
        </Col>
      </Row>
    </Spin>
  );
};

export default RiskPage;
