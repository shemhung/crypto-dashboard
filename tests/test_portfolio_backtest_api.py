from collections.abc import Generator
from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.dependencies import (
    get_market_repository,
    get_price_history_loader,
)
from backend.app.main import app


client = TestClient(app)


class FakeMarketRepository:
    """提供假的 BTC 風險歷史資料。"""

    def get_risk_history(
        self,
        **_kwargs,
    ) -> list[dict]:
        return [
            {
                "score_time": datetime(
                    2025,
                    1,
                    1,
                ),
                "total_risk": 0.10,
            },
            {
                "score_time": datetime(
                    2025,
                    1,
                    2,
                ),
                "total_risk": 0.50,
            },
            {
                "score_time": datetime(
                    2025,
                    1,
                    3,
                ),
                "total_risk": 0.90,
            },
            {
                "score_time": datetime(
                    2025,
                    1,
                    4,
                ),
                "total_risk": 0.50,
            },
        ]


class EmptyRiskRepository:
    """模擬資料庫找不到風險資料。"""

    def get_risk_history(
        self,
        **_kwargs,
    ) -> list[dict]:
        return []


def fake_price_loader(
    **kwargs,
) -> pd.DataFrame:
    """
    模擬 Binance 歷史價格。

    BTC：
    100 → 110 → 120 → 130

    ETH：
    50 → 60 → 80 → 90
    """

    symbol = kwargs["symbol"]

    prices = {
        "BTCUSDT": [
            100.0,
            110.0,
            120.0,
            130.0,
        ],
        "ETHUSDT": [
            50.0,
            60.0,
            80.0,
            90.0,
        ],
    }

    if symbol not in prices:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                start="2025-01-01",
                periods=4,
                freq="D",
            ),
            "close": prices[symbol],
        }
    )


def empty_price_loader(
    **_kwargs,
) -> pd.DataFrame:
    """模擬 Binance 找不到交易對資料。"""

    return pd.DataFrame()


@pytest.fixture(autouse=True)
def override_dependencies(
) -> Generator[None, None, None]:
    """
    每個測試都使用假的 Repository 與價格載入器。

    因此 pytest 不會連線：
    - PostgreSQL
    - Binance
    """

    original_overrides = (
        app.dependency_overrides.copy()
    )

    app.dependency_overrides[
        get_market_repository
    ] = lambda: FakeMarketRepository()

    app.dependency_overrides[
        get_price_history_loader
    ] = lambda: fake_price_loader

    yield

    app.dependency_overrides.clear()

    app.dependency_overrides.update(
        original_overrides
    )


def create_valid_request() -> dict:
    return {
        "risk_symbol": "BTCUSDT",
        "start_date": "2025-01-01",
        "end_date": "2025-01-04",
        "daily_budget": 1000,
        "buy_min": 0,
        "buy_max": 0.2,
        "sell_min": 0.8,
        "sell_max": 1,
        "sell_pct": 0.5,
        "fee_rate": 0,
        "allocations": [
            {
                "asset": "BTC",
                "weight": 0.6,
            },
            {
                "asset": "ETH",
                "weight": 0.4,
            },
        ],
    }


def test_create_portfolio_backtest():
    response = client.post(
        "/api/v1/backtests/portfolio",
        json=create_valid_request(),
    )

    assert response.status_code == 200

    body = response.json()
    summary = body["summary"]

    assert summary[
        "total_contributed"
    ] == pytest.approx(1000.0)

    assert summary[
        "cash_balance"
    ] == pytest.approx(680.0)

    assert summary[
        "market_value"
    ] == pytest.approx(750.0)

    assert summary[
        "total_equity"
    ] == pytest.approx(1430.0)

    assert summary[
        "total_profit"
    ] == pytest.approx(430.0)

    assert summary[
        "roi_pct"
    ] == pytest.approx(43.0)

    assert summary["buy_days"] == 1
    assert summary["sell_days"] == 1
    assert summary["trade_count"] == 4

    assert len(body["equity_curve"]) == 4
    assert len(body["assets"]) == 2
    assert len(body["trades"]) == 4

    asset_names = {
        asset["asset"]
        for asset in body["assets"]
    }

    assert asset_names == {
        "BTC",
        "ETH",
    }


def test_weight_total_must_equal_one():
    request = create_valid_request()

    request["allocations"] = [
        {
            "asset": "BTC",
            "weight": 0.6,
        },
        {
            "asset": "ETH",
            "weight": 0.3,
        },
    ]

    response = client.post(
        "/api/v1/backtests/portfolio",
        json=request,
    )

    assert response.status_code == 422


def test_duplicate_assets_are_rejected():
    request = create_valid_request()

    request["allocations"] = [
        {
            "asset": "BTC",
            "weight": 0.5,
        },
        {
            "asset": "btc",
            "weight": 0.5,
        },
    ]

    response = client.post(
        "/api/v1/backtests/portfolio",
        json=request,
    )

    assert response.status_code == 422


def test_overlapping_risk_ranges_are_rejected():
    request = create_valid_request()

    request["buy_max"] = 0.8
    request["sell_min"] = 0.7

    response = client.post(
        "/api/v1/backtests/portfolio",
        json=request,
    )

    assert response.status_code == 422


def test_missing_risk_history_returns_404():
    app.dependency_overrides[
        get_market_repository
    ] = lambda: EmptyRiskRepository()

    response = client.post(
        "/api/v1/backtests/portfolio",
        json=create_valid_request(),
    )

    assert response.status_code == 404

    assert response.json()["detail"] == (
        "指定期間找不到風險資料"
    )


def test_missing_asset_price_returns_422():
    app.dependency_overrides[
        get_price_history_loader
    ] = lambda: empty_price_loader

    response = client.post(
        "/api/v1/backtests/portfolio",
        json=create_valid_request(),
    )

    assert response.status_code == 422

    assert "沒有可用的歷史價格" in (
        response.json()["detail"]
    )