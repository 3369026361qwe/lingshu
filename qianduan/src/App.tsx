import React, { Suspense, lazy } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { Layout, Menu, Spin } from 'antd';
import { DashboardOutlined, RobotOutlined, StockOutlined, AlertOutlined, ExperimentOutlined } from '@ant-design/icons';
import ErrorBoundary from './components/ErrorBoundary';

// 路由级懒加载 — 首屏只加载 Dashboard，其他页面按需加载
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const SelectionPage = lazy(() => import('./pages/SelectionPage'));
const AgentPage     = lazy(() => import('./pages/AgentPage'));
const RiskPage      = lazy(() => import('./pages/RiskPage'));
const BacktestPage  = lazy(() => import('./pages/BacktestPage'));

/** 懒加载页面的 Suspense 包装器 */
const LazyPage: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Suspense fallback={
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
      <Spin size="large" tip="加载中..." />
    </div>
  }>
    {children}
  </Suspense>
);

const { Header, Sider, Content } = Layout;

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: <Link to="/">仪表盘</Link> },
  { key: '/selection', icon: <StockOutlined />, label: <Link to="/selection">选股</Link> },
  { key: '/agents', icon: <RobotOutlined />, label: <Link to="/agents">智能体</Link> },
  { key: '/risk', icon: <AlertOutlined />, label: <Link to="/risk">风控</Link> },
  { key: '/backtest', icon: <ExperimentOutlined />, label: <Link to="/backtest">回测</Link> },
];

const App: React.FC = () => {
  const location = useLocation();
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={220} style={{ background: '#161B22', borderRight: '1px solid #21262D' }}>
        <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#D4AF37', fontSize: 22, fontWeight: 700, letterSpacing: 2 }}>
          灵枢 LingShu
        </div>
        <Menu mode="inline" selectedKeys={[location.pathname]} items={menuItems} style={{ background: 'transparent', borderRight: 0 }} theme="dark" />
      </Sider>
      <Layout>
        <Header style={{ background: '#161B22', borderBottom: '1px solid #21262D', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px' }}>
          <span className="text-muted" style={{ fontSize: 14 }}>🟢 系统正常</span>
          <span className="text-muted">v3.1.0</span>
        </Header>
        <Content style={{ padding: 24, background: '#0D1117', overflow: 'auto' }}>
          <Routes>
            <Route path="/" element={<ErrorBoundary><LazyPage><DashboardPage /></LazyPage></ErrorBoundary>} />
            <Route path="/selection" element={<ErrorBoundary><LazyPage><SelectionPage /></LazyPage></ErrorBoundary>} />
            <Route path="/agents" element={<ErrorBoundary><LazyPage><AgentPage /></LazyPage></ErrorBoundary>} />
            <Route path="/risk" element={<ErrorBoundary><LazyPage><RiskPage /></LazyPage></ErrorBoundary>} />
            <Route path="/backtest" element={<ErrorBoundary><LazyPage><BacktestPage /></LazyPage></ErrorBoundary>} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
