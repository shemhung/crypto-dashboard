import pandas as pd
from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from backend.app.schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    PortfolioPointResponse,
    TradeResponse,
)
from backend.app.services.backtest_service import (
    run_backtest,
)


router = APIRouter(
    prefix="/api/v1/backtests",
    tags=["backtests"],
)


@router.post(
    "",
    response_model=BacktestResponse,
)
def create_backtest(
    request: BacktestRequest,
) -> BacktestResponse:
    """使用傳入的市場資料執行單一資產回測。"""

    market_data = pd.DataFrame(
        [
            record.model_dump()
            for record in request.records
        ]
    )

    try:
        (
            trades_df,
            portfolio_df,
            buy_days,
            sell_days,
            final_price,
        ) = run_backtest(
            df=market_data,
            buy_amount=request.buy_amount,
            buy_min=request.buy_min,
            buy_max=request.buy_max,
            sell_pct=request.sell_pct,
            sell_min=request.sell_min,
            sell_max=request.sell_max,
            start_date=request.start_date,
            fee_rate=request.fee_rate,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc

    trades = [
        TradeResponse(
            date=pd.Timestamp(
                row["Date"]
            ).to_pydatetime(),
            type=str(row["Type"]),
            price=float(row["Price"]),
            risk=float(row["Risk"]),
            value_usdt=float(row["Val_USDT"]),
            amount=float(row["Amount"]),
            fee=float(row["Fee"]),
            balance=float(row["Balance"]),
        )
        for _, row in trades_df.iterrows()
    ]

    portfolio = [
        PortfolioPointResponse(
            date=pd.Timestamp(
                row["Date"]
            ).to_pydatetime(),
            equity=float(row["Equity"]),
            invested=float(row["Invested"]),
            realized_pnl=float(
                row["Realized_PnL"]
            ),
            unrealized_pnl=float(
                row["Unrealized_PnL"]
            ),
            total_fees=float(
                row["Total_Fees"]
            ),
            avg_cost=float(row["Avg_Cost"]),
            peak_equity=float(
                row["Peak_Equity"]
            ),
        )
        for _, row in portfolio_df.iterrows()
    ]

    return BacktestResponse(
        trade_count=len(trades),
        buy_days=buy_days,
        sell_days=sell_days,
        final_price=final_price,
        trades=trades,
        portfolio=portfolio,
    )