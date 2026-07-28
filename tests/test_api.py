from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "crypto-dashboard-api",
    }


def test_calculate_risk_endpoint():
    request_body = {
        "records": [
            {
                "open_time": "2025-01-01T00:00:00",
                "close": 90000,
                "volume": 1000,
                "fear_greed": 20,
            },
            {
                "open_time": "2025-01-02T00:00:00",
                "close": 92000,
                "volume": 1200,
                "fear_greed": 40,
            },
            {
                "open_time": "2025-01-03T00:00:00",
                "close": 95000,
                "volume": 1500,
                "fear_greed": 60,
            },
        ]
    }

    response = client.post(
        "/api/v1/risk/calculate",
        json=request_body,
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["count"] == 3
    assert len(response_body["records"]) == 3

    for record in response_body["records"]:
        assert 0.0 <= record["price_risk"] <= 1.0
        assert 0.0 <= record["social_risk"] <= 1.0
        assert 0.0 <= record["total_risk"] <= 1.0


def test_calculate_risk_rejects_empty_records():
    response = client.post(
        "/api/v1/risk/calculate",
        json={
            "records": [],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "records 不可為空"


def test_calculate_risk_rejects_invalid_close_price():
    response = client.post(
        "/api/v1/risk/calculate",
        json={
            "records": [
                {
                    "open_time": "2025-01-01T00:00:00",
                    "close": -100,
                    "volume": 1000,
                    "fear_greed": 20,
                }
            ],
        },
    )

    assert response.status_code == 422