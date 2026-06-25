"""测试API层: REST端点 + WebSocket + Pydantic Schema。"""
import pytest
from fastapi.testclient import TestClient
from jiekou.server import app
from shujuku.session import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup():
    init_db(drop_all=True)


class TestHealthEndpoint:
    def test_health(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["version"] == "3.0.0"


class TestStockEndpoints:
    def test_list_stocks(self):
        r = client.get("/api/stocks")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_stock_not_found(self):
        r = client.get("/api/stocks/999999")
        assert r.status_code == 404


class TestSelectionEndpoint:
    def test_selection(self):
        r = client.get("/api/selection", params={"top_n": 5})
        assert r.status_code == 200
        assert "picks" in r.json()


class TestAgentEndpoint:
    def test_agent_reports(self):
        r = client.get("/api/agents/reports")
        assert r.status_code == 200


class TestPortfolioEndpoint:
    def test_portfolio(self):
        r = client.get("/api/portfolio")
        assert r.status_code == 200


class TestRiskEndpoint:
    def test_risk_status(self):
        r = client.get("/api/risk/status")
        assert r.status_code == 200
        data = r.json()
        assert "risk_level" in data
        assert "blocked" in data


class TestBacktestEndpoint:
    def test_backtest_run(self):
        r = client.post("/api/backtest", json={"start_date": "20260101", "end_date": "20260301", "initial_capital": "1000000"})
        assert r.status_code == 200
        assert "experiment_id" in r.json()

    def test_backtest_result(self):
        r = client.get("/api/backtest/summary")
        assert r.status_code == 200


class TestWebSocket:
    def test_ws_market_connect(self):
        with client.websocket_connect("/ws/market") as ws:
            ws.send_text("ping")
            data = ws.receive_json()
            assert data["type"] == "market"

    def test_ws_agents_connect(self):
        with client.websocket_connect("/ws/agents") as ws:
            ws.send_text("ping")
            data = ws.receive_json()
            assert data["type"] == "agent"

    def test_ws_risk_connect(self):
        with client.websocket_connect("/ws/risk") as ws:
            ws.send_text("ping")
            data = ws.receive_json()
            assert data["type"] == "risk"
