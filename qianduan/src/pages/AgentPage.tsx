import React, { useMemo } from 'react';
import { Card, Tag, Spin, Empty, Row, Col, Descriptions, Timeline } from 'antd';
import { RobotOutlined, ReloadOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import api from '../utils/api';
import { signalColor, signalLabel, formatDateShort } from '../utils/format';
import { POLL_INTERVAL } from '../utils/constants';
import { AGENT_LABELS, AGENT_COLORS } from '../types/api';
import useAPI from '../hooks/useAPI';
import useInterval from '../hooks/useInterval';
import useWebSocket from '../hooks/useWebSocket';
import type { AgentReport, AgentId } from '../types/api';

// ── Mock 兜底 ────────────────────────────────────────
const MOCK_AGENTS: AgentReport[] = [
  { agent_id: 'macro', timestamp: '2026-01-15T09:30:00', signal: 'bullish', confidence: '0.78', reasoning: '[示例] PMI连续3月扩张，制造业景气度提升。CPI温和上行，消费复苏信号明确。建议超配顺周期板块。央行维持宽松基调，流动性充裕支撑估值。' },
  { agent_id: 'sector', timestamp: '2026-01-15T09:35:00', signal: 'bullish', confidence: '0.82', reasoning: '[示例] 新能源板块资金持续流入，光伏和锂电是当前最强赛道。AI算力产业链景气度上行，半导体设备国产化加速。消费电子周期触底回升。' },
  { agent_id: 'stock', timestamp: '2026-01-15T09:40:00', signal: 'bullish', confidence: '0.71', reasoning: '[示例] 300750宁德时代：Q1出货量超预期，海外市占率提升至36%。002594比亚迪：全系车型降价刺激销量，Q1净利润同比+45%。估值处于历史中枢，安全边际充足。' },
  { agent_id: 'sentiment', timestamp: '2026-01-15T09:45:00', signal: 'neutral', confidence: '0.63', reasoning: '[示例] 社交平台讨论热度处于均值水平，散户情绪中性偏乐观。机构研报集中推荐电力设备和汽车板块。无重大负面舆情事件。' },
  { agent_id: 'risk', timestamp: '2026-01-15T09:50:00', signal: 'bearish', confidence: '0.85', reasoning: '[示例] 北向资金连续3日净流出累计82亿。全市场波动率略有上升，但仍处于历史低位。建议总仓位控制在80%以下，回避高估值成长股。' },
];

const AgentPage: React.FC = () => {
  const { data: reports, loading, refetch } =
    useAPI<AgentReport[]>(() => api.get('/api/agents/reports', { limit: 5 }));

  const { data: wsAgent } = useWebSocket('/ws/agents');

  useInterval(() => refetch(), POLL_INTERVAL.agents);

  const agents = useMemo(() => {
    if (reports?.length) return reports;
    return MOCK_AGENTS;
  }, [reports]);

  // 信号分布饼图
  const signalChart = useMemo(() => {
    const counts: Record<string, number> = {};
    agents.forEach(a => { counts[a.signal] = (counts[a.signal] || 0) + 1; });
    return {
      tooltip: { trigger: 'item' },
      series: [{
        type: 'pie',
        radius: ['50%', '80%'],
        center: ['50%', '50%'],
        itemStyle: { borderRadius: 4, borderColor: '#0D1117', borderWidth: 3 },
        label: { color: '#9AA0A6' },
        data: [
          { value: counts.bullish || 0, name: '看多', itemStyle: { color: '#2ECC71' } },
          { value: counts.neutral || 0, name: '中性', itemStyle: { color: '#9AA0A6' } },
          { value: counts.bearish || 0, name: '看空', itemStyle: { color: '#FF475C' } },
        ],
      }],
    };
  }, [agents]);

  return (
    <Spin spinning={loading}>
      <Row gutter={[16, 16]}>
        {/* ── 信号分布 ── */}
        <Col span={8}>
          <Card title="📊 信号分布" style={{ background: '#161B22', border: '1px solid #21262D', height: 320 }}>
            {agents.length ? <ReactECharts option={signalChart} style={{ height: 240 }} /> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />}
          </Card>
        </Col>

        {/* ── 实时推送 ── */}
        <Col span={8}>
          <Card title="📡 实时推送" style={{ background: '#161B22', border: '1px solid #21262D', height: 320 }}>
            {wsAgent ? (
              <div style={{ padding: '8px 12px', background: '#0D1117', borderRadius: 6, borderLeft: `3px solid ${AGENT_COLORS[wsAgent.agent_id as AgentId] || '#D4AF37'}` }}>
                <Tag color={AGENT_COLORS[wsAgent.agent_id as AgentId] || '#D4AF37'}>{AGENT_LABELS[wsAgent.agent_id as AgentId] || wsAgent.agent_id}</Tag>
                <div style={{ color: '#E8EAED', marginTop: 8, fontSize: 13 }}>{wsAgent.reasoning}</div>
                <div style={{ color: '#9AA0A6', fontSize: 12, marginTop: 4 }}>置信度: {(parseFloat(wsAgent.confidence || '0') * 100).toFixed(0)}%</div>
              </div>
            ) : (
              <Empty description="等待 WebSocket 推送..." image={Empty.PRESENTED_IMAGE_SIMPLE} />
            )}
          </Card>
        </Col>

        {/* ── 摘要 ── */}
        <Col span={8}>
          <Card
            title={<span><RobotOutlined style={{ color: '#D4AF37' }} /> Agent 摘要</span>}
            extra={<ReloadOutlined onClick={() => refetch()} style={{ cursor: 'pointer', color: '#9AA0A6' }} />}
            style={{ background: '#161B22', border: '1px solid #21262D', height: 320, overflow: 'auto' }}
          >
            {agents.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} /> : (
              <Timeline items={agents.map(a => ({
                color: signalColor(a.signal),
                children: (
                  <div key={a.agent_id}>
                    <Tag color={AGENT_COLORS[a.agent_id as AgentId]}>{AGENT_LABELS[a.agent_id as AgentId] || a.agent_id}</Tag>
                    <span style={{ color: '#E8EAED', fontSize: 13 }}>{a.reasoning.slice(0, 60)}...</span>
                    <span style={{ float: 'right', color: signalColor(a.signal), fontSize: 12 }}>{signalLabel(a.signal)}</span>
                  </div>
                ),
              }))} />
            )}
          </Card>
        </Col>

        {/* ── 详细报告 ── */}
        <Col span={24}>
          <Card title="📝 详细分析报告" className="card-dark">
            {agents.map((a, i) => (
              <Card
                key={i}
                size="small"
                style={{
                  background: '#0D1117',
                  border: `1px solid #21262D`,
                  borderLeft: `4px solid ${AGENT_COLORS[a.agent_id as AgentId] || '#D4AF37'}`,
                  marginBottom: 12,
                }}
                title={
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span>
                      <Tag color={AGENT_COLORS[a.agent_id as AgentId]}>{AGENT_LABELS[a.agent_id as AgentId] || a.agent_id}</Tag>
                      <span style={{ color: '#E8EAED' }}>{a.timestamp ? formatDateShort(a.timestamp) : ''}</span>
                    </span>
                    <span>
                      <Tag color={signalColor(a.signal)}>{signalLabel(a.signal)}</Tag>
                      <span style={{ color: '#9AA0A6', fontSize: 12 }}>{(parseFloat(a.confidence) * 100).toFixed(0)}% 置信</span>
                    </span>
                  </div>
                }
              >
                <Descriptions column={1} size="small" colon={false} labelStyle={{ color: '#9AA0A6', fontSize: 12 }} contentStyle={{ color: '#E8EAED', fontSize: 14 }}>
                  <Descriptions.Item label="分析结论">{a.reasoning}</Descriptions.Item>
                </Descriptions>
              </Card>
            ))}
          </Card>
        </Col>
      </Row>
    </Spin>
  );
};

export default AgentPage;
