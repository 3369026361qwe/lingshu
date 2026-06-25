import React, { useMemo, useState } from 'react';
import { Card, Table, InputNumber, Tag, Spin, Empty, Button, Space } from 'antd';
import { ReloadOutlined, TrophyOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import api from '../utils/api';
import { POLL_INTERVAL } from '../utils/constants';
import useAPI from '../hooks/useAPI';
import useInterval from '../hooks/useInterval';
import ScoreBar from '../components/ScoreBar';
import type { SelectionResponse, StockPick } from '../types/api';

const SelectionPage: React.FC = () => {
  const [topN, setTopN] = useState(30);

  const { data, loading, refetch } = useAPI<SelectionResponse>(
    () => api.get('/api/selection', { top_n: topN }),
    [topN],
  );

  useInterval(() => refetch(), POLL_INTERVAL.agents);

  const picks: StockPick[] = data?.picks ?? [];
  const dateLabel = data?.date ?? '—';

  // 得分分布图
  const scoreChart = useMemo(() => {
    const scores = picks.map(p => p.score).sort((a, b) => b - a);
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', data: scores.map((_, i) => i + 1), name: '排名' },
      yAxis: { type: 'value', name: '综合得分', splitLine: { lineStyle: { color: '#21262D' } } },
      series: [{
        type: 'bar',
        data: scores,
        itemStyle: {
          color: (params: any) => {
            const rank = params.dataIndex + 1;
            if (rank <= 5) return '#D4AF37';
            if (rank <= 10) return '#4198FF';
            return '#00D4AA';
          },
          borderRadius: [4, 4, 0, 0],
        },
        emphasis: { itemStyle: { color: '#D4AF37' } },
      }],
    };
  }, [picks]);

  return (
    <Spin spinning={loading}>
      {/* ── 控制栏 ── */}
      <Card size="small" style={{ background: '#161B22', border: '1px solid #21262D', marginBottom: 16 }}>
        <Space>
          <span style={{ color: '#9AA0A6' }}>返回数量</span>
          <InputNumber min={5} max={100} value={topN} onChange={v => v && setTopN(v)} style={{ width: 80 }} />
          <Button icon={<ReloadOutlined />} onClick={refetch} size="small">刷新</Button>
          <span style={{ color: '#6B7280', fontSize: 12, marginLeft: 16 }}>
            {data ? `数据日期: ${dateLabel} | 共 ${data.count} 条` : '—'}
          </span>
        </Space>
      </Card>

      {/* ── 得分分布 + 表格 ── */}
      {!picks.length && !loading ? (
        <Card style={{ background: '#161B22', border: '1px solid #21262D' }}>
          <Empty description="暂无选股结果，请确保已有因子数据和选股计算" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        </Card>
      ) : (
        <>
          <Card title={<span><TrophyOutlined style={{ color: '#D4AF37' }} /> 综合得分分布</span>} style={{ background: '#161B22', border: '1px solid #21262D', marginBottom: 16 }}>
            <ReactECharts option={scoreChart} style={{ height: 280 }} />
          </Card>

          <Card title={`选股排名 Top ${topN}`} style={{ background: '#161B22', border: '1px solid #21262D' }}>
            <Table
              dataSource={picks}
              rowKey="code"
              size="small"
              pagination={{ pageSize: 20, showSizeChanger: false }}
              columns={[
                {
                  title: '排名', dataIndex: 'rank', key: 'rank', width: 64,
                  render: (v: number) => {
                    const emoji = v === 1 ? '🥇' : v === 2 ? '🥈' : v === 3 ? '🥉' : v;
                    return <span style={{ color: v <= 3 ? '#D4AF37' : '#9AA0A6', fontWeight: v <= 3 ? 700 : 400, fontSize: v <= 3 ? 16 : 14 }}>{emoji}</span>;
                  },
                },
                {
                  title: '代码', dataIndex: 'code', key: 'code', width: 96,
                  render: (v: string) => <span style={{ color: '#4198FF', fontFamily: 'monospace', fontWeight: 600 }}>{v}</span>,
                },
                {
                  title: '综合得分', dataIndex: 'score', key: 'score',
                  render: (v: number) => <ScoreBar label="" value={v / 100} color={v >= 90 ? '#D4AF37' : v >= 80 ? '#2ECC71' : '#4198FF'} maxWidth={1} />,
                  sorter: (a: StockPick, b: StockPick) => b.score - a.score,
                  defaultSortOrder: 'descend',
                },
              ]}
              style={{ background: 'transparent' }}
            />
          </Card>
        </>
      )}
    </Spin>
  );
};

export default SelectionPage;
