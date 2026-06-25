import React, { useState, useCallback, useMemo } from 'react';
import { Row, Col, Card, Table, Button, Input, DatePicker, InputNumber, Spin, Empty, Descriptions, Statistic, Alert, Space } from 'antd';
import { PlayCircleOutlined, ReloadOutlined, ExperimentOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import dayjs from 'dayjs';
import api from '../utils/api';
import { formatMoney, formatPct } from '../utils/format';
import { POLL_INTERVAL } from '../utils/constants';
import useAPI from '../hooks/useAPI';
import useInterval from '../hooks/useInterval';
import type { BacktestResult, BacktestSummary } from '../types/api';

const BacktestPage: React.FC = () => {
  const [config, setConfig] = useState({ start: '20260101', end: '20260615', capital: 1_000_000 });
  const [running, setRunning] = useState(false);
  const [lastResult, setLastResult] = useState<BacktestResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // 回测摘要 (轮询刷新)
  const { data: summary, refetch: refetchSummary } = useAPI<BacktestSummary>(
    () => api.get('/api/backtest/summary'),
  );
  useInterval(() => refetchSummary(), POLL_INTERVAL.backtest);

  // 启动回测
  const runBacktest = useCallback(async () => {
    setRunning(true);
    setRunError(null);
    try {
      const result = await api.post<BacktestResult>('/api/backtest', {
        start_date: config.start,
        end_date: config.end,
        initial_capital: String(config.capital),
      });
      setLastResult(result);
      refetchSummary();
    } catch (err) {
      setRunError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }, [config]);

  // 权益曲线 (使用 summary 中的多日 equity_curve)
  const equityChart = useMemo(() => {
    const curve = summary?.equity_curve;
    if (!curve || curve.length < 2) return null;
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', data: curve.map(p => p.date.slice(5)), axisLabel: { rotate: 30 } },
      yAxis: { type: 'value', splitLine: { lineStyle: { color: '#21262D' } } },
      series: [{
        name: '策略权益',
        type: 'line',
        data: curve.map(p => p.value),
        smooth: true,
        lineStyle: { color: '#D4AF37', width: 2 },
        itemStyle: { color: '#D4AF37' },
        areaStyle: { color: 'rgba(212,175,55,0.1)' },
      }],
    };
  }, [summary]);

  return (
    <Row gutter={[16, 16]}>
      {/* ── 回测配置 ── */}
      <Col span={8}>
        <Card title={<span><ExperimentOutlined /> 回测配置</span>} style={{ background: '#161B22', border: '1px solid #21262D' }}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <div>
              <div style={{ color: '#9AA0A6', marginBottom: 4, fontSize: 13 }}>开始日期</div>
              <DatePicker
                value={dayjs(config.start)}
                onChange={d => d && setConfig(c => ({ ...c, start: d.format('YYYYMMDD') }))}
                style={{ width: '100%', background: '#0D1117', color: '#E8EAED' }}
              />
            </div>
            <div>
              <div style={{ color: '#9AA0A6', marginBottom: 4, fontSize: 13 }}>结束日期</div>
              <DatePicker
                value={dayjs(config.end)}
                onChange={d => d && setConfig(c => ({ ...c, end: d.format('YYYYMMDD') }))}
                style={{ width: '100%', background: '#0D1117', color: '#E8EAED' }}
              />
            </div>
            <div>
              <div style={{ color: '#9AA0A6', marginBottom: 4, fontSize: 13 }}>初始资金</div>
              <InputNumber
                value={config.capital}
                onChange={v => v && setConfig(c => ({ ...c, capital: v! }))}
                min={10_000} max={100_000_000} step={100_000}
                style={{ width: '100%' }}
                formatter={v => formatMoney(Number(v), 0)}
              />
            </div>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={runBacktest}
              loading={running}
              block
              style={{ background: '#D4AF37', borderColor: '#D4AF37', color: '#0D1117' }}
            >
              {running ? '回测运行中...' : '运行回测'}
            </Button>
            {runError && <Alert type="error" message={runError} showIcon closable onClose={() => setRunError(null)} />}
            {lastResult && (
              <Alert type="success" message={`回测完成 (ID: ${lastResult.experiment_id}) — 耗时 ${lastResult.elapsed_seconds.toFixed(1)}s`} showIcon />
            )}
          </Space>
        </Card>
      </Col>

      {/* ── 绩效指标 ── */}
      <Col span={16}>
        <Card
          title="📊 回测绩效"
          extra={<ReloadOutlined onClick={() => refetchSummary()} style={{ cursor: 'pointer', color: '#9AA0A6' }} />}
          style={{ background: '#161B22', border: '1px solid #21262D', marginBottom: 16 }}
        >
          {summary ? (
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Statistic title="总收益率" value={formatPct(summary.total_return)} valueStyle={{ color: summary.total_return >= 0 ? '#2ECC71' : '#FF475C', fontSize: 22 }} />
              </Col>
              <Col span={6}>
                <Statistic title="夏普比率" value={summary.sharpe.toFixed(2)} valueStyle={{ color: summary.sharpe >= 1 ? '#2ECC71' : '#FF8C42', fontSize: 22 }} />
              </Col>
              <Col span={6}>
                <Statistic title="最大回撤" value={formatPct(summary.max_drawdown)} valueStyle={{ color: '#FF475C', fontSize: 22 }} />
              </Col>
              <Col span={6}>
                <Statistic title="回测天数" value={summary.total_days} valueStyle={{ color: '#E8EAED', fontSize: 22 }} />
              </Col>
            </Row>
          ) : (
            <Empty description="暂无回测数据，请先运行一次回测" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          )}
        </Card>

        {/* ── 权益曲线 ── */}
        <Card title="📈 权益曲线" style={{ background: '#161B22', border: '1px solid #21262D' }}>
          {equityChart ? (
            <ReactECharts option={equityChart} style={{ height: 280 }} />
          ) : (
            <Empty description="运行回测后显示权益曲线" image={Empty.PRESENTED_IMAGE_SIMPLE} />
          )}
        </Card>
      </Col>

      {/* ── 指标说明 ── */}
      <Col span={24}>
        <Card title="📋 指标说明" size="small" style={{ background: '#161B22', border: '1px solid #21262D' }}>
          <Descriptions column={3} size="small" colon={false} labelStyle={{ color: '#9AA0A6' }} contentStyle={{ color: '#E8EAED' }}>
            <Descriptions.Item label="夏普比率">年化收益率与年化波动率之比，≥1 为良好，≥2 为优秀</Descriptions.Item>
            <Descriptions.Item label="最大回撤">峰值到谷底的最大跌幅，反映尾部风险</Descriptions.Item>
            <Descriptions.Item label="IC / IR">因子 IC 均值 / IC 标准差，衡量因子稳定性</Descriptions.Item>
          </Descriptions>
        </Card>
      </Col>
    </Row>
  );
};

export default BacktestPage;
