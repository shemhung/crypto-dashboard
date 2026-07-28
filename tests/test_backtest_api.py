from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def create_request_body() -> dict:
    """建立固定的回測 API 測試資料。"""

    return {
        "records": [
            {
                "open_time": (
                    "2025-01-01T00:00:00"
                ),
                "asset_price": 100.0,
                "total_risk": 0.10,
            },
            {
                "open_time": (
                    "2025-01-02T00:00:00"
                ),
                "asset_price": 110.0,
                "total_risk": 0.50,
            },
            {
                "open_time": (
                    "2025-01-03T00:00:00"
                ),
                "asset_price": 120.0,
                "total_risk": 0.90,
            },
            {
                "open_time": (
                    "2025-01-04T00:00:00"
                ),
                "asset_price": 130.0,
                "total_risk": 0.50,
            },
        ],
        "buy_amount": 1000.0,
        "buy_min": 0.0,
        "buy_max": 0.2,
        "sell_pct": 0.5,
        "sell_min": 0.8,
        "sell_max": 1.0,
        "start_date": "2025-01-01",
        "fee_rate": 0.001,
    }


def test_create_backtest():
    response = client.post(
        "/api/v1/backtests",
        json=create_request_body(),
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["trade_count"] == 2
    assert response_body["buy_days"] == 1
    assert response_body["sell_days"] == 1
    assert response_body["final_price"] == 130.0

    assert len(response_body["trades"]) == 2
    assert len(response_body["portfolio"]) == 4

    assert (
        response_body["trades"][0]["type"]
        == "BUY"
    )

    assert (
        response_body["trades"][1]["type"]
        == "SELL"
    )


def test_backtest_rejects_empty_records():
    request_body = create_request_body()
    request_body["records"] = []

    response = client.post(
        "/api/v1/backtests",
        json=request_body,
    )

    assert response.status_code == 422


def test_backtest_rejects_invalid_buy_range():
    request_body = create_request_body()

    request_body["buy_min"] = 0.5
    request_body["buy_max"] = 0.2

    response = client.post(
        "/api/v1/backtests",
        json=request_body,
    )

    assert response.status_code == 422


def test_backtest_rejects_invalid_sell_percentage():
    request_body = create_request_body()
    request_body["sell_pct"] = 1.5

    response = client.post(
        "/api/v1/backtests",
        json=request_body,
    )

    assert response.status_code == 422


def test_backtest_rejects_invalid_risk_value():
    request_body = create_request_body()
    request_body["records"][0][
        "total_risk"
    ] = -0.1

    response = client.post(
        "/api/v1/backtests",
        json=request_body,
    )

    assert response.status_code == 422