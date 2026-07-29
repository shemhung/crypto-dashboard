from dataclasses import dataclass
from datetime import date
from math import isclose, isfinite

import pandas as pd


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """投資組合回測結果。"""

    summary: dict[str, float | int]
    equity_curve: pd.DataFrame
    asset_results: pd.DataFrame
    trades: pd.DataFrame


def run_portfolio_backtest(
    market_data: pd.DataFrame,
    allocations: dict[str, float],
    daily_budget: float,
    buy_min: float,
    buy_max: float,
    sell_min: float,
    sell_max: float,
    sell_pct: float,
    start_date: date,
    end_date: date | None = None,
    fee_rate: float = 0.001,
) -> PortfolioBacktestResult:
    """
    執行多幣種投資組合回測。

    market_data 必須包含：
    - open_time
    - total_risk
    - allocations 中每個幣種的價格欄位

    例如：
    open_time | total_risk | BTC | ETH | SOL
    """

    if not allocations:
        raise ValueError("至少要選擇一個幣種")

    if daily_budget <= 0:
        raise ValueError("daily_budget 必須大於 0")

    if not 0 <= fee_rate <= 0.1:
        raise ValueError("fee_rate 必須介於 0 與 0.1 之間")

    if not 0 < sell_pct <= 1:
        raise ValueError("sell_pct 必須大於 0 且不超過 1")

    if not 0 <= buy_min < buy_max <= 1:
        raise ValueError("買入風險區間設定錯誤")

    if not 0 <= sell_min < sell_max <= 1:
        raise ValueError("賣出風險區間設定錯誤")

    if buy_max > sell_min:
        raise ValueError("買入與賣出風險區間不可重疊")

    if end_date is not None and start_date > end_date:
        raise ValueError("start_date 不可晚於 end_date")

    weight_total = sum(allocations.values())

    if not isclose(
        weight_total,
        1.0,
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise ValueError("幣種權重總和必須等於 1.0")

    for asset, weight in allocations.items():
        if weight <= 0:
            raise ValueError(
                f"{asset} 的權重必須大於 0"
            )

    required_columns = {
        "open_time",
        "total_risk",
        *allocations.keys(),
    }

    missing_columns = (
        required_columns -
        set(market_data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"市場資料缺少欄位：{sorted(missing_columns)}"
        )

    data = market_data.copy(deep=True)

    data["open_time"] = pd.to_datetime(
        data["open_time"],
        errors="coerce",
    )

    data["total_risk"] = pd.to_numeric(
        data["total_risk"],
        errors="coerce",
    )

    assets = list(allocations.keys())

    for asset in assets:
        data[asset] = pd.to_numeric(
            data[asset],
            errors="coerce",
        )

    data = (
        data
        .dropna(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # 只能使用過去已知價格，不能向後填值。
    data[assets] = data[assets].ffill()

    data = data[
        data["open_time"].dt.date >= start_date
    ]

    if end_date is not None:
        data = data[
            data["open_time"].dt.date <= end_date
        ]

    balances = {
        asset: 0.0
        for asset in assets
    }

    cost_bases = {
        asset: 0.0
        for asset in assets
    }

    asset_contributed = {
        asset: 0.0
        for asset in assets
    }

    asset_realized_pnl = {
        asset: 0.0
        for asset in assets
    }

    asset_fees = {
        asset: 0.0
        for asset in assets
    }

    latest_prices = {
        asset: 0.0
        for asset in assets
    }

    cash_balance = 0.0
    total_contributed = 0.0
    total_fees = 0.0

    buy_days = 0
    sell_days = 0
    peak_equity = 0.0

    trade_history: list[dict] = []
    equity_history: list[dict] = []

    for _, row in data.iterrows():
        current_date = row["open_time"]
        risk_value = row["total_risk"]

        if (
            pd.isna(risk_value) or
            not isfinite(float(risk_value))
        ):
            continue

        risk = float(risk_value)

        # 更新目前可使用的最新價格。
        for asset in assets:
            price_value = row[asset]

            if (
                pd.notna(price_value) and
                isfinite(float(price_value)) and
                float(price_value) > 0
            ):
                latest_prices[asset] = float(
                    price_value
                )

        # 低風險：投入新的每日預算並按權重買入。
        if buy_min <= risk < buy_max:
            buy_days += 1

            total_contributed += daily_budget
            cash_balance += daily_budget

            for asset, weight in allocations.items():
                price = latest_prices[asset]

                # 該幣種目前還沒有價格時，
                # 對應預算保留為現金。
                if price <= 0:
                    continue

                allocated_budget = (
                    daily_budget * weight
                )

                fee = (
                    allocated_budget * fee_rate
                )

                net_purchase_value = (
                    allocated_budget - fee
                )

                amount = (
                    net_purchase_value / price
                )

                cash_balance -= allocated_budget

                balances[asset] += amount

                # 買入成本包含買入手續費。
                cost_bases[asset] += (
                    allocated_budget
                )

                asset_contributed[asset] += (
                    allocated_budget
                )

                asset_fees[asset] += fee
                total_fees += fee

                trade_history.append(
                    {
                        "Date": current_date,
                        "Asset": asset,
                        "Type": "BUY",
                        "Price": price,
                        "Risk": risk,
                        "Amount": amount,
                        "Value_USDT": allocated_budget,
                        "Fee": fee,
                    }
                )

        # 高風險：賣出每個幣種目前持倉的一部分。
        elif sell_min <= risk <= sell_max:
            sell_days += 1

            for asset in assets:
                price = latest_prices[asset]
                current_balance = balances[asset]

                if (
                    price <= 0 or
                    current_balance <= 0
                ):
                    continue

                amount_to_sell = (
                    current_balance * sell_pct
                )

                average_cost = (
                    cost_bases[asset] /
                    current_balance
                )

                removed_cost = (
                    average_cost *
                    amount_to_sell
                )

                gross_value = (
                    amount_to_sell * price
                )

                fee = gross_value * fee_rate
                net_value = gross_value - fee

                balances[asset] -= amount_to_sell

                cost_bases[asset] = max(
                    0.0,
                    cost_bases[asset] -
                    removed_cost,
                )

                cash_balance += net_value

                realized_profit = (
                    net_value - removed_cost
                )

                asset_realized_pnl[
                    asset
                ] += realized_profit

                asset_fees[asset] += fee
                total_fees += fee

                trade_history.append(
                    {
                        "Date": current_date,
                        "Asset": asset,
                        "Type": "SELL",
                        "Price": price,
                        "Risk": risk,
                        "Amount": amount_to_sell,
                        "Value_USDT": net_value,
                        "Fee": fee,
                    }
                )

        if abs(cash_balance) < 1e-10:
            cash_balance = 0.0

        total_market_value = sum(
            balances[asset] *
            latest_prices[asset]
            for asset in assets
        )

        remaining_cost_basis = sum(
            cost_bases.values()
        )

        total_realized_pnl = sum(
            asset_realized_pnl.values()
        )

        total_unrealized_pnl = (
            total_market_value -
            remaining_cost_basis
        )

        # 正確的總權益：
        # 賣出後現金 + 尚未賣出的持倉市值。
        total_equity = (
            cash_balance +
            total_market_value
        )

        peak_equity = max(
            peak_equity,
            total_equity,
        )

        drawdown_pct = (
            (
                total_equity -
                peak_equity
            )
            / peak_equity
            * 100
            if peak_equity > 0
            else 0.0
        )

        equity_history.append(
            {
                "Date": current_date,
                "Cash": cash_balance,
                "Market_Value": (
                    total_market_value
                ),
                "Equity": total_equity,
                "Contributed": (
                    total_contributed
                ),
                "Realized_PnL": (
                    total_realized_pnl
                ),
                "Unrealized_PnL": (
                    total_unrealized_pnl
                ),
                "Total_Fees": total_fees,
                "Peak_Equity": peak_equity,
                "Drawdown_Pct": drawdown_pct,
            }
        )

    equity_curve = pd.DataFrame(
        equity_history
    )

    trades = pd.DataFrame(
        trade_history
    )

    if equity_curve.empty:
        final_equity = 0.0
        final_market_value = 0.0
        final_cash = 0.0
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        mdd_pct = 0.0

    else:
        final_row = equity_curve.iloc[-1]

        final_equity = float(
            final_row["Equity"]
        )

        final_market_value = float(
            final_row["Market_Value"]
        )

        final_cash = float(
            final_row["Cash"]
        )

        realized_pnl = float(
            final_row["Realized_PnL"]
        )

        unrealized_pnl = float(
            final_row["Unrealized_PnL"]
        )

        mdd_pct = float(
            equity_curve[
                "Drawdown_Pct"
            ].min()
        )

    total_profit = (
        final_equity -
        total_contributed
    )

    roi_pct = (
        total_profit /
        total_contributed *
        100
        if total_contributed > 0
        else 0.0
    )

    asset_result_rows = []

    for asset, weight in allocations.items():
        last_price = latest_prices[asset]

        market_value = (
            balances[asset] *
            last_price
        )

        unrealized = (
            market_value -
            cost_bases[asset]
        )

        asset_profit = (
            asset_realized_pnl[asset] +
            unrealized
        )

        contributed = (
            asset_contributed[asset]
        )

        asset_roi_pct = (
            asset_profit /
            contributed *
            100
            if contributed > 0
            else 0.0
        )

        asset_result_rows.append(
            {
                "Asset": asset,
                "Weight": weight,
                "Balance": balances[asset],
                "Last_Price": last_price,
                "Market_Value": market_value,
                "Cost_Basis": cost_bases[asset],
                "Contributed": contributed,
                "Realized_PnL": (
                    asset_realized_pnl[
                        asset
                    ]
                ),
                "Unrealized_PnL": unrealized,
                "Profit": asset_profit,
                "ROI_Pct": asset_roi_pct,
                "Fees": asset_fees[asset],
            }
        )

    summary: dict[str, float | int] = {
        "total_contributed": (
            total_contributed
        ),
        "total_equity": final_equity,
        "total_profit": total_profit,
        "roi_pct": roi_pct,
        "mdd_pct": mdd_pct,
        "buy_days": buy_days,
        "sell_days": sell_days,
        "trade_count": len(trades),
        "cash_balance": final_cash,
        "market_value": final_market_value,
        "total_fees": total_fees,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
    }

    return PortfolioBacktestResult(
        summary=summary,
        equity_curve=equity_curve,
        asset_results=pd.DataFrame(
            asset_result_rows
        ),
        trades=trades,
    )