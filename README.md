# 灵枢 LingShu

基于 **LLM 多智能体** 与 **图神经网络** 的 A 股量化投资系统。

## 核心创新

- 🧠 **LLM 多智能体协作** — 5 个专业 Agent 并行分析（宏观/赛道/个股/舆情/风控）
- 🕸️ **GNN 产业链建模** — 5000+ 节点异构图捕捉产业链传导
- 📐 **卡尔曼滤波动态权重** — 因子权重在线自适应估计
- 🛡️ **四层风控体系** — 熔断器 + 仓位管理 + VaR 监控 + AI 风控 Agent

## 架构

```
shuju (数据) → yinzi (因子) → zhinengti (智能体) + tushenjing (图神经)
                                    ↓
                               juece (决策) → zhixing (执行)
                                    ↓
                               jiekou (API) → qianduan (前端)
```

## 快速启动

```bash
# 安装依赖
pip install -e .

# 启动 API 服务
uvicorn jiekou.server:app --host 0.0.0.0 --port 8000 --reload

# 运行测试
python -m pytest tests/ -v
```

## 文档

详见 [设计文档](lingshulianghuasheji/灵枢量化系统-设计文档.md)
