from datetime import date

import pandas as pd
import pytest

from backend.app.services.portfolio_backtest_service import (
    run_portfolio_backtest,
)


def create_sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                start="2025-01-01",
                periods=4,
                freq="D",
            ),
            "total_risk": [
                0.10,  # 買入
                0.50,  # 不操作
                0.90,  # 賣出 50%
                0.50,  # 不操作
            ],
            "BTC": [
                100.0,
                110.0,
                120.0,
                130.0,
            ],
            "ETH": [
                50.0,
                60.0,
                80.0,
                90.0,
            ],
        }
    )


def run_sample_backtest(
    fee_rate: float = 0.0,
):
    return run_portfolio_backtest(
        market_data=create_sample_data(),
        allocations={
            "BTC": 0.6,
            "ETH": 0.4,
        },
        daily_budget=1000.0,
        buy_min=0.0,
        buy_max=0.2,
        sell_min=0.8,
        sell_max=1.0,
        sell_pct=0.5,
        start_date=date(2025, 1, 1),
        fee_rate=fee_rate,
    )


def test_portfolio_allocation_and_roi():
    result = run_sample_backtest()

    summary = result.summary

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


def test_asset_weights_control_buy_amount():
    result = run_sample_backtest()

    buy_trades = result.trades[
        result.trades["Type"] == "BUY"
    ]

    btc_buy = buy_trades[
        buy_trades["Asset"] == "BTC"
    ].iloc[0]

    eth_buy = buy_trades[
        buy_trades["Asset"] == "ETH"
    ].iloc[0]

    assert btc_buy[
        "Value_USDT"
    ] == pytest.approx(600.0)

    assert eth_buy[
        "Value_USDT"
    ] == pytest.approx(400.0)


def test_weight_total_must_equal_one():
    with pytest.raises(
        ValueError,
        match="權重總和",
    ):
        run_portfolio_backtest(
            market_data=create_sample_data(),
            allocations={
                "BTC": 0.6,
                "ETH": 0.3,
            },
            daily_budget=1000.0,
            buy_min=0.0,
            buy_max=0.2,
            sell_min=0.8,
            sell_max=1.0,
            sell_pct=0.5,
            start_date=date(2025, 1, 1),
        )


def test_fees_are_recorded():
    result = run_sample_backtest(
        fee_rate=0.001,
    )

    assert result.summary[
        "total_fees"
    ] > 0

    assert result.trades[
        "Fee"
    ].gt(0).all()


def test_max_drawdown_is_calculated():
    market_data = pd.DataFrame(
        {
            "open_time": pd.date_range(
                start="2025-01-01",
                periods=3,
                freq="D",
            ),
            "total_risk": [
                0.1,
                0.5,
                0.5,
            ],
            "BTC": [
                100.0,
                50.0,
                50.0,
            ],
            "ETH": [
                50.0,
                25.0,
                25.0,
            ],
        }
    )

    result = run_portfolio_backtest(
        market_data=market_data,
        allocations={
            "BTC": 0.5,
            "ETH": 0.5,
        },
        daily_budget=1000.0,
        buy_min=0.0,
        buy_max=0.2,
        sell_min=0.8,
        sell_max=1.0,
        sell_pct=0.5,
        start_date=date(2025, 1, 1),
        fee_rate=0.0,
    )

    assert result.summary[
        "mdd_pct"
    ] == pytest.approx(-50.0)


def test_original_dataframe_is_not_modified():
    market_data = create_sample_data()
    original_data = market_data.copy(
        deep=True
    )

    run_portfolio_backtest(
        market_data=market_data,
        allocations={
            "BTC": 0.6,
            "ETH": 0.4,
        },
        daily_budget=1000.0,
        buy_min=0.0,
        buy_max=0.2,
        sell_min=0.8,
        sell_max=1.0,
        sell_pct=0.5,
        start_date=date(2025, 1, 1),
    )

    pd.testing.assert_frame_equal(
        market_data,
        original_data,
    )