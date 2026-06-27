import React, { useMemo } from 'react';
import { Row, Col, Card, Table, Tag, Button, Spin, Empty, Alert, Statistic, Space, Descriptions } from 'antd';
import { PlayCircleOutlined, ReloadOutlined, RiseOutlined, FallOutlined, DollarOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import api from '../utils/api';
import { formatMoney, formatPct, riskColor } from '../utils/format';
import { POLL_INTERVAL } from '../utils/constants';
import useAPI from '../hooks/useAPI';
import useInterval from '../hooks/useInterval';
import ScoreBar from '../components/ScoreBar';
import type { TradePipeline, TradeOrder } from '../types/api';

const TradingPage: React.FC = () => {
  const { data: pipeline, loading, refetch } = useAPI<TradePipeline>(
    () => api.get('/api/trade/pipeline', { top_n: 15 }),
  );

  const { data: history } = useAPI<TradeOrder[]>(
    () => api.get('/api/trade/history', { limit: 10 }),
  );

  useInterval(() => refetch(), POLL_INTERVAL.agents);

  // 组合配置饼图
  const allocChart = useMemo(() => {
    const pf = pipeline?.portfolio ?? [];
    if (!pf.length) return null;
    return {
      tooltip: { trigger: 'item', formatter: '{b}: {c}%' },
      series: [{
        type: 'pie', radius: ['50%', '80%'], center: ['50%', '55%'],
        itemStyle: { borderRadius: 3, borderColor: '#0D1117', borderWidth: 2 },
        label: { color: '#9AA0A6', fontSize: 10, formatter: '{b}' },
        data: pf.map((p, i) => ({
          value: +(p.weight * 100).toFixed(1),
          name: p.code,
          itemStyle: { color: ['#D4AF37', '#2ECC71', '#4198FF', '#FF8C42', '#00D4AA', '#FF475C', '#A78BFA', '#34D399'][i % 8] },
        })),
      }],
    };
  }, [pipeline]);

  // 风险仪表盘
  const riskGauge = useMemo(() => {
    const score = pipeline?.risk?.score ?? 0;
    return {
      series: [{
        type: 'gauge', startAngle: 210, endAngle: -30, center: ['50%', '60%'], radius: '85%',
        min: 0, max: 10,
        axisLine: { lineStyle: { width: 16, color: [[0.3, '#2ECC71'], [0.6, '#FF8C42'], [1, '#FF475C']] } },
        pointer: { length: '65%', width: 5 },
        detail: { valueAnimation: true, formatter: '{value}', color: '#E8EAED', fontSize: 20 },
        data: [{ value: score, name: '风险分' }],
      }],
    };
  }, [pipeline]);

  const buys = pipeline?.trades?.buys ?? [];
  const sells = pipeline?.trades?.sells ?? [];
  const riskBlocked = pipeline?.risk?.blocked;

  return (
    <Spin spinning={loading}>
      <Row gutter={[16, 16]}>
        {/* ── 控制栏 ── */}
        <Col span={24}>
          <Card className="card-dark" size="small">
            <Space>
              <Button type="primary" icon={<PlayCircleOutlined />} style={{ background: '#D4AF37', borderColor: '#D4AF37', color: '#0D1117' }}
                onClick={async () => {
                  await api.post('/api/trade/execute');
                  refetch();
                }}>
                执行调仓
              </Button>
              <Button icon={<ReloadOutlined />} onClick={refetch}>刷新</Button>
              <span className="text-muted">
                {pipeline ? `数据日期: ${pipeline.date} | 资金: ${formatMoney(pipeline.capital, 0)}` : '加载中...'}
              </span>
            </Space>
            {riskBlocked && (
              <Alert type="error" message="⛔ 风控拦截 — 当前风险过高，调仓已被阻止" showIcon style={{ marginTop: 8 }} />
            )}
          </Card>
        </Col>

        {/* ── 选股信号 ── */}
        <Col span={8}>
          <Card title="🎯 选股信号 Top-15" className="card-dark">
            {pipeline?.stocks?.length ? (
              pipeline.stocks.slice(0, 10).map(s => (
                <ScoreBar key={s.code} label={s.code} value={s.score} color={s.score >= 0.8 ? '#D4AF37' : s.score >= 0.5 ? '#2ECC71' : '#4198FF'} maxWidth={1} />
              ))
            ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />}
          </Card>
        </Col>

        {/* ── 调仓指令 ── */}
        <Col span={8}>
          <Card
            title={<span><RiseOutlined style={{ color: '#2ECC71' }} /> 买入 ({buys.length})</span>}
            className="card-dark" style={{ marginBottom: 16 }}
          >
            {buys.length ? (
              <Table dataSource={buys} rowKey="code" size="small" pagination={false}
                columns={[
                  { title: '代码', dataIndex: 'code', key: 'code', render: (v: string) => <span className="text-gold" style={{ fontFamily: 'monospace' }}>{v}</span> },
                  { title: '权重', dataIndex: 'weight', key: 'weight', render: (v: number) => formatPct(v) },
                  { title: '金额', dataIndex: 'amount', key: 'amount', render: (v: number) => <span className="text-green">{formatMoney(v)}</span> },
                ]}
              />
            ) : <Empty description="无买入" image={Empty.PRESENTED_IMAGE_SIMPLE} />}
          </Card>
          <Card
            title={<span><FallOutlined style={{ color: '#FF475C' }} /> 卖出 ({sells.length})</span>}
            className="card-dark"
          >
            {sells.length ? (
              <Table dataSource={sells} rowKey="code" size="small" pagination={false}
                columns={[
                  { title: '代码', dataIndex: 'code', key: 'code', render: (v: string) => <span className="text-gold" style={{ fontFamily: 'monospace' }}>{v}</span> },
                  { title: '权重', dataIndex: 'weight', key: 'weight', render: (v: number) => formatPct(v) },
                  { title: '金额', dataIndex: 'amount', key: 'amount', render: (v: number) => <span className="text-red">{formatMoney(v)}</span> },
                ]}
              />
            ) : <Empty description="无卖出" image={Empty.PRESENTED_IMAGE_SIMPLE} />}
          </Card>
        </Col>

        {/* ── 组合配置 + 风险 ── */}
        <Col span={8}>
          <Card title="📊 组合配置" className="card-dark" style={{ marginBottom: 16 }}>
            {allocChart ? <ReactECharts option={allocChart} style={{ height: 200 }} />
              : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />}
          </Card>
          <Card title="⚠ 风险概览" className="card-dark">
            {riskGauge ? <ReactECharts option={riskGauge} style={{ height: 160 }} />
              : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />}
            {pipeline?.risk && (
              <Descriptions column={1} size="small" colon={false} labelStyle={{ color: '#9AA0A6' }} contentStyle={{ color: '#E8EAED' }}>
                <Descriptions.Item label="风控建议">{pipeline.risk.advice || '正常'}</Descriptions.Item>
              </Descriptions>
            )}
          </Card>
        </Col>

        {/* ── 交易历史 ── */}
        <Col span={24}>
          <Card title="📜 交易历史" className="card-dark">
            <Table dataSource={history ?? []} rowKey={(r, i) => `${r.code}-${i}`} size="small" pagination={{ pageSize: 10 }}
              columns={[
                { title: '时间', dataIndex: 'time', key: 'time', render: (v: string) => <span className="text-muted">{v?.slice(0, 19)}</span> },
                { title: '代码', dataIndex: 'code', key: 'code', render: (v: string) => <span style={{ fontFamily: 'monospace', color: '#D4AF37' }}>{v}</span> },
                { title: '方向', dataIndex: 'direction', key: 'direction', render: (v: string) => <Tag color={v === 'BUY' ? '#2ECC71' : '#FF475C'}>{v === 'BUY' ? '买入' : '卖出'}</Tag> },
                { title: '数量', dataIndex: 'amount', key: 'amount', render: (v: number) => formatMoney(v) },
                { title: '价格', dataIndex: 'price', key: 'price', render: (v: number) => v > 0 ? v.toFixed(2) : '—' },
                { title: '状态', dataIndex: 'status', key: 'status', render: (v: string) => <Tag>{v}</Tag> },
              ]}
            />
          </Card>
        </Col>
      </Row>
    </Spin>
  );
};

export default TradingPage;
