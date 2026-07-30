from datetime import date, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from backend.app.dependencies import get_market_repository
from backend.app.main import app


class FakeMarketRepository:
    """測試用 Repository，不會連接真正的資料庫。"""

    def __init__(
        self,
        latest_record: dict[str, Any] | None = None,
        history_records: list[dict[str, Any]] | None = None,
        error: SQLAlchemyError | None = None,
    ) -> None:
        self.latest_record = latest_record
        self.history_records = history_records or []
        self.error = error

        # 保存 API 傳進 Repository 的參數，供測試確認
        self.latest_symbol: str | None = None
        self.history_arguments: tuple[str, int] | None = None

    def get_latest_risk(
        self,
        symbol: str = "BTCUSDT",
    ) -> dict[str, Any] | None:
        self.latest_symbol = symbol

        if self.error is not None:
            raise self.error

        return self.latest_record

    def get_risk_history(
        self,
        symbol: str = "BTCUSDT",
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.history_arguments = (
            symbol,
            start_date,
            end_date,
            limit,
        )

        if self.error is not None:
            raise self.error

        records = self.history_records

        if limit is not None:
            records = records[-limit:]

        return records


@pytest.fixture
def client():
    """
    每個測試開始與結束時清除 dependency override，
    避免不同測試互相影響。
    """

    app.dependency_overrides.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def use_fake_repository(
    repository: FakeMarketRepository,
) -> None:
    """將正式 Repository 替換成測試用 Repository。"""

    app.dependency_overrides[
        get_market_repository
    ] = lambda: repository


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "crypto-dashboard-api",
    }


def test_calculate_risk_endpoint(client: TestClient):
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


def test_calculate_risk_rejects_empty_records(
    client: TestClient,
):
    response = client.post(
        "/api/v1/risk/calculate",
        json={
            "records": [],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "records 不可為空"
    )


def test_calculate_risk_rejects_invalid_close_price(
    client: TestClient,
):
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


def test_get_latest_risk(client: TestClient):
    repository = FakeMarketRepository(
        latest_record={
            "symbol": "BTCUSDT",
            "score_time": datetime(
                2026,
                7,
                28,
                0,
                0,
            ),
            "price": 118000.0,
            "total_risk": 0.42,
            "price_risk": 0.30,
            "social_risk": 0.42,
            "risk_level": "HODL",
        }
    )

    use_fake_repository(repository)

    response = client.get(
        "/api/v1/risk/latest",
        params={
            "symbol": "btcusdt",
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["symbol"] == "BTCUSDT"
    assert response_body["price"] == 118000.0
    assert response_body["total_risk"] == 0.42
    assert response_body["risk_level"] == "HODL"

    # 確認 API 有把 symbol 轉成大寫
    assert repository.latest_symbol == "BTCUSDT"


def test_get_risk_history(client: TestClient):
    repository = FakeMarketRepository(
        history_records=[
            {
                "symbol": "BTCUSDT",
                "score_time": datetime(
                    2026,
                    7,
                    27,
                ),
                "price": 116000.0,
                "total_risk": 0.38,
                "price_risk": 0.25,
                "social_risk": 0.38,
                "risk_level": "BUY",
            },
            {
                "symbol": "BTCUSDT",
                "score_time": datetime(
                    2026,
                    7,
                    28,
                ),
                "price": 118000.0,
                "total_risk": 0.42,
                "price_risk": 0.30,
                "social_risk": 0.42,
                "risk_level": "HODL",
            },
        ]
    )

    use_fake_repository(repository)

    response = client.get(
        "/api/v1/risk/history",
        params={
            "symbol": "btcusdt",
            "limit": 2,
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["count"] == 2
    assert len(response_body["records"]) == 2

    assert (
        response_body["records"][0]["price"]
        == 116000.0
    )

    assert (
        response_body["records"][1]["price"]
        == 118000.0
    )

    assert repository.history_arguments == (
        "BTCUSDT",
        None,
        None,
        2,
    )


def test_get_latest_risk_returns_404_when_not_found(
    client: TestClient,
):
    repository = FakeMarketRepository(
        latest_record=None,
    )

    use_fake_repository(repository)

    response = client.get(
        "/api/v1/risk/latest",
        params={
            "symbol": "ETHUSDT",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "找不到指定資產的風險資料"
    )


def test_get_latest_risk_returns_503_on_database_error(
    client: TestClient,
):
    repository = FakeMarketRepository(
        error=SQLAlchemyError(
            "database unavailable"
        )
    )

    use_fake_repository(repository)

    response = client.get(
        "/api/v1/risk/latest"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "資料庫目前無法使用"
    )


def test_get_risk_history_rejects_invalid_limit(
    client: TestClient,
):
    repository = FakeMarketRepository()

    use_fake_repository(repository)

    response = client.get(
        "/api/v1/risk/history",
        params={
            "limit": 0,
        },
    )

    assert response.status_code == 422


def test_get_risk_history_with_date_range(
    client: TestClient,
):
    repository = FakeMarketRepository(
        history_records=[
            {
                "symbol": "BTCUSDT",
                "score_time": datetime(
                    2017,
                    8,
                    17,
                ),
                "price": 4300.0,
                "total_risk": 0.20,
                "price_risk": 0.10,
                "social_risk": 0.20,
                "risk_level": "BUY",
            },
            {
                "symbol": "BTCUSDT",
                "score_time": datetime(
                    2026,
                    7,
                    29,
                ),
                "price": 118000.0,
                "total_risk": 0.45,
                "price_risk": 0.30,
                "social_risk": 0.45,
                "risk_level": "HODL",
            },
        ]
    )

    use_fake_repository(repository)

    response = client.get(
        "/api/v1/risk/history",
        params={
            "symbol": "btcusdt",
            "start_date": "2017-08-17",
            "end_date": "2026-07-29",
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["count"] == 2

    assert repository.history_arguments == (
        "BTCUSDT",
        date(2017, 8, 17),
        date(2026, 7, 29),
        None,
    )


def test_get_risk_history_rejects_invalid_date_range(
    client: TestClient,
):
    repository = FakeMarketRepository()

    use_fake_repository(repository)

    response = client.get(
        "/api/v1/risk/history",
        params={
            "start_date": "2026-07-29",
            "end_date": "2017-08-17",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "start_date 不可晚於 end_date"
    )
    