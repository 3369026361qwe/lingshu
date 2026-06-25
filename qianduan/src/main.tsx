import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ConfigProvider, theme } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import App from './App';
import './styles/theme.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider locale={zhCN} theme={{algorithm: theme.darkAlgorithm, token: {colorPrimary: '#D4AF37', borderRadius: 6, colorBgBase: '#0D1117'}}}>
      <BrowserRouter><App /></BrowserRouter>
    </ConfigProvider>
  </React.StrictMode>
);
