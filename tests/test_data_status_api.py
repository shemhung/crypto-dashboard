from collections.abc import Generator
from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from backend.app.dependencies import (
    get_market_api_checker,
    get_market_repository,
)
from backend.app.main import app


client = TestClient(app)


class FreshRiskRepository:
    """模擬資料庫有最新風險資料。"""

    def get_risk_history(
        self,
        **_kwargs,
    ) -> list[dict]:
        now = datetime.now(
            timezone.utc
        )

        return [
            {
                "score_time": (
                    now -
                    timedelta(days=2)
                ),
                "total_risk": 0.35,
            },
            {
                "score_time": (
                    now -
                    timedelta(hours=12)
                ),
                "total_risk": 0.40,
            },
        ]


class StaleRiskRepository:
    """模擬風險資料超過 48 小時。"""

    def get_risk_history(
        self,
        **_kwargs,
    ) -> list[dict]:
        now = datetime.now(
            timezone.utc
        )

        return [
            {
                "score_time": (
                    now -
                    timedelta(days=5)
                ),
                "total_risk": 0.35,
            },
        ]


class EmptyRiskRepository:
    """模擬資料庫沒有風險資料。"""

    def get_risk_history(
        self,
        **_kwargs,
    ) -> list[dict]:
        return []


class UnavailableRepository:
    """模擬資料庫無法連線。"""

    def get_risk_history(
        self,
        **_kwargs,
    ) -> list[dict]:
        raise SQLAlchemyError(
            "database unavailable"
        )


@pytest.fixture(autouse=True)
def override_dependencies(
) -> Generator[None, None, None]:
    original_overrides = (
        app.dependency_overrides.copy()
    )

    app.dependency_overrides[
        get_market_repository
    ] = lambda: FreshRiskRepository()

    app.dependency_overrides[
        get_market_api_checker
    ] = lambda: (
        lambda: True
    )

    yield

    app.dependency_overrides.clear()

    app.dependency_overrides.update(
        original_overrides
    )


def test_data_status_is_healthy():
    response = client.get(
        "/api/v1/data-status"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"

    assert body[
        "database"
    ]["connected"] is True

    assert body[
        "market_api"
    ]["available"] is True

    assert body[
        "risk_data"
    ]["symbol"] == "BTCUSDT"

    assert body[
        "risk_data"
    ]["record_count"] == 2

    assert body[
        "risk_data"
    ]["is_stale"] is False

    assert (
        body["risk_data"]["age_hours"]
        < 48
    )


def test_stale_data_is_degraded():
    app.dependency_overrides[
        get_market_repository
    ] = lambda: StaleRiskRepository()

    response = client.get(
        "/api/v1/data-status"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "degraded"

    assert body[
        "database"
    ]["connected"] is True

    assert body[
        "risk_data"
    ]["is_stale"] is True


def test_empty_data_is_degraded():
    app.dependency_overrides[
        get_market_repository
    ] = lambda: EmptyRiskRepository()

    response = client.get(
        "/api/v1/data-status"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "degraded"

    assert body[
        "risk_data"
    ]["record_count"] == 0

    assert body[
        "risk_data"
    ]["latest_time"] is None

    assert body[
        "risk_data"
    ]["is_stale"] is True


def test_market_api_failure_is_degraded():
    app.dependency_overrides[
        get_market_api_checker
    ] = lambda: (
        lambda: False
    )

    response = client.get(
        "/api/v1/data-status"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "degraded"

    assert body[
        "database"
    ]["connected"] is True

    assert body[
        "market_api"
    ]["available"] is False


def test_database_failure_is_unavailable():
    app.dependency_overrides[
        get_market_repository
    ] = lambda: UnavailableRepository()

    response = client.get(
        "/api/v1/data-status"
    )

    assert response.status_code == 200

    body = response.json()

    assert body[
        "status"
    ] == "unavailable"

    assert body[
        "database"
    ]["connected"] is False

    assert body[
        "risk_data"
    ]["record_count"] == 0

    assert body[
        "risk_data"
    ]["is_stale"] is True