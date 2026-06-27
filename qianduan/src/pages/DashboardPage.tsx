import React, { useEffect, useMemo } from 'react';
import { Row, Col, Card, Table, Tag, Spin, Empty } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, RobotOutlined, ReloadOutlined, ShareAltOutlined } from '@ant-design/icons';
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
import type { SelectionResponse, AgentReport, RiskStatus, EquityResponse, FactorWeightsResponse, MarketData, GNNGraphData } from '../types/api';
import { FALLBACK_EQUITY, MOCK_PICKS, MOCK_AGENTS, FALLBACK_WEIGHTS, FALLBACK_MARKET } from '../utils/mockData';


/** 单指数 StatCard 的渲染器 */
const IndexCard: React.FC<{ idx: { index: string; value: string; change: string; up?: boolean } }> = ({ idx }) => {
  const up = idx.up ?? !idx.change.startsWith('-');
  return (
    <StatCard
      title={`📊 ${idx.index}`}
      value={idx.value}
      prefix={up ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
      suffix={<span style={{ fontSize: 14, color: up ? 'var(--accent-green)' : 'var(--accent-red)' }}>{idx.change}</span>}
      accent={up ? '#2ECC71' : '#FF475C'}
      footer="实时行情"
    />
  );
};


const DashboardPage: React.FC = () => {
  // ── 全局状态 ──────────────────────────────────────
  const marketStore = useMarketStore();
  const setMarket = marketStore.setMarket;
  const setRisk = useRiskStore((s) => s.setRisk);
  const setSelection = useSelectionStore((s) => s.setSelection);

  // ── 数据源 ────────────────────────────────────────
  const { data: wsMarket } = useWebSocket('/ws/market');
  const { data: wsRisk } = useWebSocket('/ws/risk');

  const { data: selection, loading: selLoading, refetch: refetchSel } =
    useAPI<SelectionResponse>(() => api.get('/api/selection', { top_n: 10 }));

  const { data: agentReports, loading: agentsLoading, refetch: refetchAgents } =
    useAPI<AgentReport[]>(() => api.get('/api/agents/reports', { limit: 3 }));

  const { data: riskStatus, loading: riskLoading, refetch: refetchRisk } =
    useAPI<RiskStatus>(() => api.get('/api/risk/status'));

  const { data: equityData } =
    useAPI<EquityResponse>(() => api.get('/api/equity'));

  const { data: factorData } =
    useAPI<FactorWeightsResponse>(() => api.get('/api/factors/weights'));

  const { data: gnnData } =
    useAPI<GNNGraphData>(() => api.get('/api/gnn/graph', { top_n: 40 }));

  // ── 定时刷新 ──────────────────────────────────────
  useInterval(() => { refetchSel(); refetchAgents(); refetchRisk(); }, POLL_INTERVAL.agents);

  // ── 同步到全局 store ─────────────────────────────
  useEffect(() => {
    if (wsMarket?.data) setMarket(wsMarket.data as unknown as Record<string, unknown>);
  }, [wsMarket, setMarket]);

  useEffect(() => {
    if (wsRisk) setRisk(wsRisk as unknown as Record<string, unknown>);
  }, [wsRisk, setRisk]);

  useEffect(() => {
    if (selection) setSelection(selection);
  }, [selection, setSelection]);

  // ── 市场数据（真实 → Mock 兜底）──────────────────
  const market: MarketData = useMemo(() => {
    const raw = (wsMarket?.data ?? FALLBACK_MARKET) as Record<string, unknown>;
    return {
      latest_date: (raw.latest_date as string) ?? '',
      stock_count: (raw.stock_count as number) ?? 0,
      avg_change_pct: (raw.avg_change_pct as number) ?? 0,
      csi300: (raw.csi300 as MarketData['csi300']) ?? FALLBACK_MARKET.csi300,
      csi500: (raw.csi500 as MarketData['csi500']) ?? FALLBACK_MARKET.csi500,
      chinext: (raw.chinext as MarketData['chinext']) ?? FALLBACK_MARKET.chinext,
      updated_at: (raw.updated_at as string) ?? '',
    };
  }, [wsMarket]);

  // ── Mock 兜底合并 ─────────────────────────────────
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

  // ── 权益曲线 ──────────────────────────────────────
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

  // ── 因子权重 ──────────────────────────────────────
  const factors = useMemo(() => {
    if (factorData?.weights?.length) return factorData.weights;
    return FALLBACK_WEIGHTS;
  }, [factorData]);

  // ── GNN 力导向图 ─────────────────────────────────
  const gnnOption = useMemo(() => {
    const nodes = gnnData?.nodes ?? [];
    const edges = gnnData?.edges ?? [];
    if (!nodes.length) return null;

    return {
      tooltip: {
        formatter: (p: any) => {
          if (p.dataType === 'node') return `${p.name}<br/>GNN得分: ${p.value?.toFixed(4) ?? '—'}`;
          return `${p.data.source} ↔ ${p.data.target}`;
        },
      },
      legend: {
        data: ['得分 > 0.7', '0.4–0.7', '< 0.4'],
        textStyle: { color: '#9AA0A6', fontSize: 11 },
        top: 0,
      },
      series: [{
        type: 'graph',
        layout: 'force',
        roam: true,
        draggable: true,
        force: { repulsion: 200, edgeLength: [30, 120], gravity: 0.1 },
        categories: [
          { name: '得分 > 0.7', itemStyle: { color: '#D4AF37' } },
          { name: '0.4–0.7', itemStyle: { color: '#2ECC71' } },
          { name: '< 0.4', itemStyle: { color: '#4198FF' } },
        ],
        data: nodes.map(n => {
          const cat = n.symbolSize >= 28 ? '得分 > 0.7' : n.symbolSize >= 20 ? '0.4–0.7' : '< 0.4';
          return { name: n.id, value: n.score, symbolSize: n.symbolSize, category: cat };
        }),
        edges: edges.map(e => ({ source: e.source, target: e.target })),
        lineStyle: { color: '#21262D', curveness: 0.1, opacity: 0.4 },
        label: {
          show: true,
          position: 'right',
          fontSize: 9,
          color: '#9AA0A6',
          formatter: (p: any) => p.name,
        },
        emphasis: {
          focus: 'adjacency',
          lineStyle: { width: 2, opacity: 0.8 },
          itemStyle: { shadowBlur: 10, shadowColor: 'rgba(212,175,55,0.5)' },
        },
      }],
    };
  }, [gnnData]);

  // ── 渲染 ──────────────────────────────────────────
  return (
    <Row gutter={[16, 16]}>
      {/* ── 三指数市场概览 ── */}
      <Col span={6}><IndexCard idx={market.csi300} /></Col>
      <Col span={6}><IndexCard idx={market.csi500} /></Col>
      <Col span={6}><IndexCard idx={market.chinext} /></Col>
      <Col span={6}>
        <Card size="small" loading={riskLoading} className="card-dark" style={{ border: `1px solid ${riskColor(riskLevel)}` }}>
          <div className="text-muted" style={{ marginBottom: 8 }}>⚠ 风险等级</div>
          <RiskBadge level={riskLevel as RiskStatus['risk_level']} />
          <div className="text-muted" style={{ marginTop: 8 }}>
            熔断器: {breakerState} | VaR 95%: {var95}
          </div>
        </Card>
      </Col>

      {/* ── AI 智能体洞察 ── */}
      <Col span={12}>
        <Card
          title={<span><RobotOutlined style={{ color: '#D4AF37' }} /> AI智能体今日洞察</span>}
          extra={<ReloadOutlined onClick={() => refetchAgents()} className="icon-click" />}
          className="card-dark"
        >
          <Spin spinning={agentsLoading}>
            {agents.length === 0 && !agentsLoading ? (
              <Empty description="暂无智能体报告" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              agents.map((a, i) => (
                <div key={i} className="agent-row" style={{ borderLeft: `3px solid ${signalColor(a.signal)}` }}>
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
          extra={<span className="text-muted">{selection?.date ? `数据日期: ${selection.date}` : 'Mock 数据'}</span>}
          className="card-dark"
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
                  { title: '排名', dataIndex: 'rank', key: 'rank', width: 56, render: (v: number) => <span className="text-gold" style={{ fontWeight: 700 }}>{v}</span> },
                  { title: '代码', dataIndex: 'code', key: 'code', width: 80, render: (v: string) => <span className="text-gold">{v}</span> },
                  { title: '综合得分', dataIndex: 'score', key: 'score', render: (v: number) => <ScoreBar label="" value={v / 100} color="#2ECC71" maxWidth={1} />, sorter: (a: any, b: any) => b.score - a.score },
                ]}
              />
            )}
          </Spin>
        </Card>
      </Col>

      {/* ── 收益曲线 ── */}
      <Col span={14}>
        <Card title="📈 收益曲线 (累计 vs 沪深300)" className="card-dark">
          <ReactECharts option={equityOption} style={{ height: 300 }} />
        </Card>
      </Col>

      {/* ── GNN 产业链图 ── */}
      <Col span={10}>
        <Card
          title={<span><ShareAltOutlined style={{ color: '#D4AF37' }} /> GNN 产业链图</span>}
          extra={<span className="text-muted">{gnnData ? `${gnnData.node_count} 节点, ${gnnData.edge_count} 边` : '加载中...'}</span>}
          className="card-dark"
        >
          {gnnOption ? (
            <ReactECharts option={gnnOption} style={{ height: 340 }} />
          ) : gnnData?.error ? (
            <div className="flex-center" style={{ height: 300 }}>
              <Empty description={gnnData.error} image={Empty.PRESENTED_IMAGE_SIMPLE} />
            </div>
          ) : (
            <div className="flex-center" style={{ height: 300 }}>
              <Spin tip="加载产业链数据..." />
            </div>
          )}
        </Card>
      </Col>

      {/* ── 因子权重 ── */}
      <Col span={24}>
        <Card
          title="⚖️ 因子权重"
          extra={<span className="text-muted">{factorData?.source === 'live' ? '📡 实时' : '📋 Mock'}</span>}
          className="card-dark"
        >
          <Row gutter={[16, 8]}>
            {factors.map(f => {
              const barColor = f.source === 'synthetic' ? '#D4AF37' : f.source === 'kalman' ? '#4198FF' : '#6B7280';
              const sourceTag = f.source === 'synthetic' ? ' 🤖' : f.source === 'kalman' ? '' : ' *';
              return (
                <Col span={8} key={f.name}>
                  <ScoreBar label={f.name + sourceTag} value={f.weight} color={barColor} maxWidth={0.25} />
                </Col>
              );
            })}
          </Row>
          <div className="text-muted" style={{ marginTop: 8 }}>
            🤖 GNN/Agent = 模型合成权重 | 其他 = 卡尔曼滤波 | * = Mock 兜底
          </div>
        </Card>
      </Col>
    </Row>
  );
};

export default DashboardPage;
