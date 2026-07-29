import pandas as pd

from backend.app.services.risk_service import (
    compute_rainbow_risk,
    compute_risk,
)


def create_sample_market_data() -> pd.DataFrame:
    """建立測試用市場資料，不需要呼叫外部 API。"""

    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                start="2025-01-01",
                periods=5,
                freq="D",
            ),
            "close": [
                90_000.0,
                92_000.0,
                95_000.0,
                93_000.0,
                98_000.0,
            ],
            "volume": [
                1000.0,
                1200.0,
                1500.0,
                1100.0,
                1800.0,
            ],
            "fear_greed": [
                20,
                40,
                60,
                50,
                80,
            ],
        }
    )


def test_compute_rainbow_risk_stays_between_zero_and_one():
    test_prices = [
        0.01,
        100.0,
        10_000.0,
        100_000.0,
        1_000_000.0,
    ]

    for price in test_prices:
        result = compute_rainbow_risk(price)

        assert 0.0 <= result <= 1.0


def test_compute_risk_adds_expected_columns():
    market_data = create_sample_market_data()

    result = compute_risk(market_data)

    expected_columns = {
        "price_risk",
        "social_risk",
        "total_risk",
    }

    assert expected_columns.issubset(result.columns)


def test_compute_risk_values_stay_between_zero_and_one():
    market_data = create_sample_market_data()

    result = compute_risk(market_data)

    risk_columns = [
        "price_risk",
        "social_risk",
        "total_risk",
    ]

    for column in risk_columns:
        assert result[column].notna().all()
        assert result[column].between(0.0, 1.0).all()


def test_compute_risk_keeps_same_number_of_rows():
    market_data = create_sample_market_data()

    result = compute_risk(market_data)

    assert len(result) == len(market_data)


def test_compute_risk_works_without_social_columns():
    market_data = pd.DataFrame(
        {
            "open_time": pd.date_range(
                start="2025-01-01",
                periods=3,
                freq="D",
            ),
            "close": [
                90_000.0,
                95_000.0,
                100_000.0,
            ],
        }
    )

    result = compute_risk(market_data)

    assert "total_risk" in result.columns
    assert result["total_risk"].notna().all()
    assert result["total_risk"].between(0.0, 1.0).all()