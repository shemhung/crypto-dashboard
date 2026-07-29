from datetime import date

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    buy_amount: float,
    buy_min: float,
    buy_max: float,
    sell_pct: float,
    sell_min: float,
    sell_max: float,
    start_date: date,
    fee_rate: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, float]:
    """
    執行單一資產回測。

    df 必須包含：
    - open_time
    - asset_price
    - total_risk
    """

    required_columns = {
        "open_time",
        "asset_price",
        "total_risk",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"回測資料缺少欄位：{sorted(missing_columns)}"
        )

    df_test = df.copy()

    df_test["open_time"] = pd.to_datetime(
        df_test["open_time"]
    )

    df_test = df_test[
        df_test["open_time"].dt.date >= start_date
    ]

    if df_test.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            0,
            0,
            0.0,
        )

    asset_balance = 0.0
    total_invested = 0.0
    realized_pnl = 0.0
    total_fees = 0.0

    trade_history = []
    portfolio_history = []

    buy_days = 0
    sell_days = 0

    for _, row in df_test.iterrows():
        price = row["asset_price"]
        risk = row["total_risk"]
        current_date = row["open_time"]

        if pd.isna(price) or price <= 0:
            continue

        action = None
        trade_value = 0.0
        trade_amount = 0.0
        fee_amount = 0.0

        # 買入
        if buy_min <= risk < buy_max:
            buy_days += 1
            action = "BUY"

            fee_amount = buy_amount * fee_rate
            net_investment = buy_amount - fee_amount
            trade_amount = net_investment / price

            asset_balance += trade_amount
            total_invested += buy_amount
            total_fees += fee_amount

            trade_value = buy_amount

        # 賣出
        elif sell_min <= risk <= sell_max:
            sell_days += 1

            if asset_balance > 0:
                action = "SELL"

                amount_to_sell = asset_balance * sell_pct

                if amount_to_sell > 0:
                    gross_value = amount_to_sell * price
                    fee_amount = gross_value * fee_rate
                    net_value = gross_value - fee_amount

                    average_cost = (
                        total_invested / asset_balance
                        if asset_balance > 0
                        else 0.0
                    )

                    cost_of_sold = (
                        amount_to_sell * average_cost
                    )

                    asset_balance -= amount_to_sell
                    total_invested -= cost_of_sold
                    realized_pnl += (
                        net_value - cost_of_sold
                    )
                    total_fees += fee_amount

                    trade_value = net_value
                    trade_amount = amount_to_sell

        market_value = asset_balance * price
        unrealized_pnl = market_value - total_invested
        total_equity = market_value + realized_pnl

        current_average_cost = (
            total_invested / asset_balance
            if asset_balance > 0
            else 0.0
        )

        if action is not None:
            trade_history.append(
                {
                    "Date": current_date,
                    "Type": action,
                    "Price": price,
                    "Risk": risk,
                    "Val_USDT": trade_value,
                    "Amount": trade_amount,
                    "Fee": fee_amount,
                    "Balance": asset_balance,
                }
            )

        previous_peak = (
            portfolio_history[-1]["Peak_Equity"]
            if portfolio_history
            else total_equity
        )

        peak_equity = max(
            previous_peak,
            total_equity,
        )

        portfolio_history.append(
            {
                "Date": current_date,
                "Equity": total_equity,
                "Invested": total_invested,
                "Realized_PnL": realized_pnl,
                "Unrealized_PnL": unrealized_pnl,
                "Total_Fees": total_fees,
                "Avg_Cost": current_average_cost,
                "Peak_Equity": peak_equity,
            }
        )

    final_price = (
        float(df_test.iloc[-1]["asset_price"])
        if not df_test.empty
        else 0.0
    )

    return (
        pd.DataFrame(trade_history),
        pd.DataFrame(portfolio_history),
        buy_days,
        sell_days,
        final_price,
    )