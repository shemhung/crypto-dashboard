from datetime import date

import pandas as pd
import pytest

from backend.app.services.backtest_service import run_backtest


def create_sample_backtest_data() -> pd.DataFrame:
    """建立不需要外部 API 的回測資料。"""

    return pd.DataFrame(
        {
            "open_time": pd.date_range(
                start="2025-01-01",
                periods=4,
                freq="D",
            ),
            "asset_price": [
                100.0,
                110.0,
                120.0,
                130.0,
            ],
            "total_risk": [
                0.10,  # 買入
                0.50,  # 不操作
                0.90,  # 賣出
                0.50,  # 不操作
            ],
        }
    )


def run_sample_backtest():
    """使用固定參數執行測試回測。"""

    return run_backtest(
        df=create_sample_backtest_data(),
        buy_amount=1000.0,
        buy_min=0.0,
        buy_max=0.2,
        sell_pct=0.5,
        sell_min=0.8,
        sell_max=1.01,
        start_date=date(2025, 1, 1),
        fee_rate=0.001,
    )


def test_run_backtest_creates_buy_and_sell_trades():
    trades, _, buy_days, sell_days, _ = run_sample_backtest()

    assert len(trades) == 2
    assert trades.iloc[0]["Type"] == "BUY"
    assert trades.iloc[1]["Type"] == "SELL"

    assert buy_days == 1
    assert sell_days == 1


def test_run_backtest_calculates_trading_fees():
    trades, portfolio, _, _, _ = run_sample_backtest()

    assert trades["Fee"].gt(0).all()
    assert portfolio.iloc[-1]["Total_Fees"] > 0


def test_run_backtest_returns_final_price():
    _, _, _, _, final_price = run_sample_backtest()

    assert final_price == pytest.approx(130.0)


def test_run_backtest_keeps_peak_equity_monotonic():
    _, portfolio, _, _, _ = run_sample_backtest()

    peak_equity = portfolio["Peak_Equity"]

    assert peak_equity.is_monotonic_increasing


def test_run_backtest_does_not_modify_original_dataframe():
    market_data = create_sample_backtest_data()
    original_data = market_data.copy(deep=True)

    run_backtest(
        df=market_data,
        buy_amount=1000.0,
        buy_min=0.0,
        buy_max=0.2,
        sell_pct=0.5,
        sell_min=0.8,
        sell_max=1.01,
        start_date=date(2025, 1, 1),
    )

    pd.testing.assert_frame_equal(
        market_data,
        original_data,
    )


def test_run_backtest_rejects_missing_columns():
    invalid_data = pd.DataFrame(
        {
            "open_time": pd.date_range(
                start="2025-01-01",
                periods=3,
                freq="D",
            ),
            "total_risk": [
                0.1,
                0.5,
                0.9,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="asset_price",
    ):
        run_backtest(
            df=invalid_data,
            buy_amount=1000.0,
            buy_min=0.0,
            buy_max=0.2,
            sell_pct=0.5,
            sell_min=0.8,
            sell_max=1.01,
            start_date=date(2025, 1, 1),
        )


def test_sell_signal_without_assets_does_not_create_trade():
    market_data = pd.DataFrame(
        {
            "open_time": pd.date_range(
                start="2025-01-01",
                periods=2,
                freq="D",
            ),
            "asset_price": [
                100.0,
                110.0,
            ],
            "total_risk": [
                0.90,
                0.95,
            ],
        }
    )

    trades, portfolio, _, _, _ = run_backtest(
        df=market_data,
        buy_amount=1000.0,
        buy_min=0.0,
        buy_max=0.2,
        sell_pct=0.5,
        sell_min=0.8,
        sell_max=1.01,
        start_date=date(2025, 1, 1),
    )

    assert trades.empty
    assert portfolio["Equity"].eq(0.0).all()