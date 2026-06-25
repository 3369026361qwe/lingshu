import React from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import { DashboardOutlined, RobotOutlined, StockOutlined, AlertOutlined, ExperimentOutlined, SettingOutlined } from '@ant-design/icons';
import ErrorBoundary from './components/ErrorBoundary';
import DashboardPage from './pages/DashboardPage';
import SelectionPage from './pages/SelectionPage';
import AgentPage from './pages/AgentPage';
import RiskPage from './pages/RiskPage';
import BacktestPage from './pages/BacktestPage';

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
        <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#D4AF37', fontSize: 22, fontWeight: 700, letterSpacing: 2 }}>灵枢 LingShu</div>
        <Menu mode="inline" selectedKeys={[location.pathname]} items={menuItems} style={{ background: 'transparent', borderRight: 0 }} theme="dark" />
      </Sider>
      <Layout>
        <Header style={{ background: '#161B22', borderBottom: '1px solid #21262D', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px' }}>
          <span style={{ color: '#E8EAED', fontSize: 14 }}>🟢 系统正常 | 2026-06-15</span>
          <span style={{ color: '#9AA0A6', fontSize: 12 }}>v3.0.0</span>
        </Header>
        <Content style={{ padding: 24, background: '#0D1117', overflow: 'auto' }}>
          <Routes>
            <Route path="/" element={<ErrorBoundary><DashboardPage /></ErrorBoundary>} />
            <Route path="/selection" element={<ErrorBoundary><SelectionPage /></ErrorBoundary>} />
            <Route path="/agents" element={<ErrorBoundary><AgentPage /></ErrorBoundary>} />
            <Route path="/risk" element={<ErrorBoundary><RiskPage /></ErrorBoundary>} />
            <Route path="/backtest" element={<ErrorBoundary><BacktestPage /></ErrorBoundary>} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
