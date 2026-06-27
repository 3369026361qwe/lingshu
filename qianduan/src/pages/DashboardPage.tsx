import React, { useEffect, useMemo } from 'react';
import { Row, Col, Card, Table, Tag, Spin, Empty } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, RobotOutlined, ReloadOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import api from '../utils/api';
import { signalColor, signalLabel, riskColor } from '../utils/format';
import { POLL_INTERVAL } from '../utils/constants';
import useAPI from '../hooks/useAPI';
import useInterval from '../hooks/useInterval';
import useWebSocket from '../hooks/useWebSocket';
import StatCard from '../components/StatCard';
import ScoreBar from '../components/ScoreBar';
import RiskBadge from '../components/RiskBadge';
import { useMarketStore } from '../stores/marketStore';
import { useRiskStore } from '../stores/riskStore';
import { useSelectionStore } from '../stores/selectionStore';
import type { SelectionResponse, AgentReport, RiskStatus, EquityResponse, FactorWeightsResponse } from '../types/api';
import { FALLBACK_EQUITY, MOCK_PICKS, MOCK_AGENTS, FALLBACK_WEIGHTS, FALLBACK_MARKET } from '../utils/mockData';

// ── 组件 ────────────────────────────────────────────

const DashboardPage: React.FC = () => {
  // 全局状态（跨页面共享）
  const setMarket = useMarketStore((s) => s.setMarket);
  const setRisk = useRiskStore((s) => s.setRisk);
  const setSelection = useSelectionStore((s) => s.setSelection);

  // WebSocket
  const { data: wsMarket } = useWebSocket('/ws/market');
  const { data: wsRisk } = useWebSocket('/ws/risk');

  // 选股结果
  const { data: selection, loading: selLoading, refetch: refetchSel } =
    useAPI<SelectionResponse>(() => api.get('/api/selection', { top_n: 10 }));

  // 智能体报告
  const { data: agentReports, loading: agentsLoading, refetch: refetchAgents } =
    useAPI<AgentReport[]>(() => api.get('/api/agents/reports', { limit: 3 }));

  // 风控状态
  const { data: riskStatus, loading: riskLoading, refetch: refetchRisk } =
    useAPI<RiskStatus>(() => api.get('/api/risk/status'));

  // 权益曲线
  const { data: equityData } =
    useAPI<EquityResponse>(() => api.get('/api/equity'));

  // 因子权重
  const { data: factorData } =
    useAPI<FactorWeightsResponse>(() => api.get('/api/factors/weights'));

  // 定时刷新 (30s)
  useInterval(() => { refetchSel(); refetchAgents(); refetchRisk(); }, POLL_INTERVAL.agents);

  // ── 同步到全局 store ────────────────────────────

  useEffect(() => {
    if (wsMarket?.data) setMarket(wsMarket.data as unknown as Record<string, unknown>);
  }, [wsMarket, setMarket]);

  useEffect(() => {
    if (wsRisk) setRisk(wsRisk as unknown as Record<string, unknown>);
  }, [wsRisk, setRisk]);

  useEffect(() => {
    if (selection) setSelection(selection);
  }, [selection, setSelection]);

  // ── 数据合并 (真实 → Mock 兜底) ──────────────

  const picks = useMemo(() => {
    if (selection?.picks?.length) return selection.picks;
    return MOCK_PICKS;
  }, [selection]);

  const agents: AgentReport[] = useMemo(() => {
    if (agentReports?.length) return agentReports;
    return MOCK_AGENTS;
  }, [agentReports]);

  const riskLevel = riskStatus?.risk_level ?? 'LOW';
  const breakerState = riskStatus?.breaker_state ?? 'CLOSED';
  const var95 = riskStatus?.var_95 ?? '1.2%';

  // 权益曲线图表
  const equityOption = useMemo(() => {
    const curve = equityData?.data?.length ? equityData.data : null;
    const dates = curve ? curve.map(p => p.date.slice(5)) : FALLBACK_EQUITY.dates;
    const values = curve ? curve.map(p => p.value) : FALLBACK_EQUITY.values;
    return {
      tooltip: { trigger: 'axis' },
      legend: { data: ['策略权益', '沪深300'], textStyle: { color: '#9AA0A6' } },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', data: dates },
      yAxis: { type: 'value', splitLine: { lineStyle: { color: '#21262D' } } },
      series: [
        { name: '策略权益', type: 'line', data: values, smooth: true, lineStyle: { color: '#D4AF37', width: 2 }, itemStyle: { color: '#D4AF37' } },
        { name: '沪深300', type: 'line', data: FALLBACK_EQUITY.bench, smooth: true, lineStyle: { color: '#6B7280', width: 1.5, type: 'dashed' }, itemStyle: { color: '#6B7280' } },
      ],
    };
  }, [equityData]);

  // 因子权重
  const factors = useMemo(() => {
    if (factorData?.weights?.length) return factorData.weights;
    return FALLBACK_WEIGHTS;
  }, [factorData]);

  const marketData = (wsMarket?.data ?? FALLBACK_MARKET) as Record<string, unknown>;
  const marketIndex = (marketData.index as string) ?? FALLBACK_MARKET.index;
  const marketChange = (marketData.change as string) ?? FALLBACK_MARKET.change;
  const isMarketUp = !marketChange.startsWith('-');

  // ── 渲染 ──────────────────────────────────────────

  return (
    <Row gutter={[16, 16]}>
      {/* ── 市场概览卡片 ── */}
      <Col span={6}>
        <StatCard
          title="📊 沪深300"
          value={marketIndex}
          prefix={isMarketUp ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
          suffix={<span style={{ fontSize: 14, color: isMarketUp ? '#2ECC71' : '#FF475C' }}>{marketChange}</span>}
          accent={isMarketUp ? '#2ECC71' : '#FF475C'}
          footer="实时行情"
        />
      </Col>
      <Col span={6}>
        <StatCard
          title="📈 中证500"
          value="5,821.35"
          suffix={<span style={{ fontSize: 14, color: '#2ECC71' }}>+1.15%</span>}
          accent="#2ECC71"
          prefix={<ArrowUpOutlined />}
        />
      </Col>
      <Col span={6}>
        <StatCard
          title="🚀 创业板指"
          value="1,892.45"
          suffix={<span style={{ fontSize: 14, color: '#2ECC71' }}>+2.03%</span>}
          accent="#2ECC71"
          prefix={<ArrowUpOutlined />}
        />
      </Col>
      <Col span={6}>
        <Card size="small" loading={riskLoading} style={{ background: '#161B22', border: `1px solid ${riskColor(riskLevel)}` }}>
          <div style={{ color: '#9AA0A6', fontSize: 13, marginBottom: 8 }}>⚠ 风险等级</div>
          <RiskBadge level={riskLevel as RiskStatus['risk_level']} />
          <div style={{ color: '#9AA0A6', fontSize: 12, marginTop: 8 }}>
            熔断器: {breakerState} | VaR 95%: {var95}
          </div>
        </Card>
      </Col>

      {/* ── AI 智能体洞察 ── */}
      <Col span={12}>
        <Card
          title={<span><RobotOutlined style={{ color: '#D4AF37' }} /> AI智能体今日洞察</span>}
          extra={<ReloadOutlined onClick={() => refetchAgents()} style={{ cursor: 'pointer', color: '#9AA0A6' }} />}
          style={{ background: '#161B22', border: '1px solid #21262D' }}
        >
          <Spin spinning={agentsLoading}>
            {agents.length === 0 && !agentsLoading ? (
              <Empty description="暂无智能体报告" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              agents.map((a, i) => (
                <div key={i} style={{ marginBottom: i < agents.length - 1 ? 12 : 0, padding: '8px 12px', background: '#0D1117', borderRadius: 6, borderLeft: `3px solid ${signalColor(a.signal)}` }}>
                  <Tag color={signalColor(a.signal)} style={{ marginRight: 8 }}>{a.agent_id}</Tag>
                  <span style={{ color: '#E8EAED' }}>"{a.reasoning}"</span>
                  <span style={{ float: 'right', color: signalColor(a.signal), fontSize: 13 }}>
                    {signalLabel(a.signal)} | {(parseFloat(a.confidence) * 100).toFixed(0)}%
                  </span>
                </div>
              ))
            )}
          </Spin>
        </Card>
      </Col>

      {/* ── 今日推荐 Top 10 ── */}
      <Col span={12}>
        <Card
          title="🏆 今日推荐"
          extra={
            <span style={{ color: '#9AA0A6', fontSize: 12 }}>
              {selection?.date ? `数据日期: ${selection.date}` : 'Mock 数据'}
            </span>
          }
          style={{ background: '#161B22', border: '1px solid #21262D' }}
        >
          <Spin spinning={selLoading}>
            {picks.length === 0 && !selLoading ? (
              <Empty description="暂无选股数据，请运行一次选股计算" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              <Table
                dataSource={picks}
                pagination={false}
                size="small"
                rowKey="code"
                columns={[
                  { title: '排名', dataIndex: 'rank', key: 'rank', width: 56, render: (v: number) => <span style={{ color: '#D4AF37', fontWeight: 700 }}>{v}</span> },
                  { title: '代码', dataIndex: 'code', key: 'code', width: 80, render: (v: string) => <span style={{ color: '#D4AF37' }}>{v}</span> },
                  { title: '综合得分', dataIndex: 'score', key: 'score', render: (v: number) => <ScoreBar label="" value={v / 100} color="#2ECC71" maxWidth={1} />, sorter: (a: any, b: any) => b.score - a.score },
                ]}
                style={{ background: 'transparent' }}
              />
            )}
          </Spin>
        </Card>
      </Col>

      {/* ── 收益曲线 ── */}
      <Col span={16}>
        <Card title="📈 收益曲线 (累计 vs 沪深300)" style={{ background: '#161B22', border: '1px solid #21262D' }}>
          <ReactECharts option={equityOption} style={{ height: 300 }} />
        </Card>
      </Col>

      {/* ── 因子权重 + GNN ── */}
      <Col span={8}>
        <Card title="⚖️ 因子权重" style={{ background: '#161B22', border: '1px solid #21262D', marginBottom: 16 }}>
          {factors.map(f => (
            <ScoreBar key={f.name} label={f.name} value={f.weight} color={f.name === 'GNN' || f.name === 'Agent' ? '#D4AF37' : '#4198FF'} maxWidth={0.25} />
          ))}
        </Card>
        <Card title="🕸️ GNN 产业链" style={{ background: '#161B22', border: '1px solid #21262D', height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ color: '#D4AF37', textAlign: 'center' }}>
            🔴 今日推荐 10<br />
            🔵 产业链关联 50<br />
            <span style={{ fontSize: 12, color: '#9AA0A6' }}>连线粗细 = 关系强度</span>
          </div>
        </Card>
      </Col>
    </Row>
  );
};

export default DashboardPage;
