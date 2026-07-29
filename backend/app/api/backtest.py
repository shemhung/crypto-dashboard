import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from backend.app.dependencies import (
    PriceHistoryLoader,
    get_market_repository,
    get_price_history_loader,
)
from backend.app.repositories.market_repository import (
    MarketRepository,
)
from backend.app.data_sources.binance import (
    BinanceAPIError,
)
from backend.app.schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    PortfolioAssetResultResponse,
    PortfolioBacktestRequest,
    PortfolioBacktestResponse,
    PortfolioBacktestSummaryResponse,
    PortfolioEquityPointResponse,
    PortfolioTradeResponse,
    PortfolioPointResponse,
    TradeResponse,
)
from backend.app.services.portfolio_backtest_service import (
    run_portfolio_backtest,
)
from fastapi import (
    APIRouter,
    Depends,
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

def prepare_portfolio_market_data(
    risk_records: list[dict],
    allocations: list,
    start_date,
    end_date,
    price_loader: PriceHistoryLoader,
) -> pd.DataFrame:
    """
    將風險資料與多個幣種價格合併成回測資料。

    最終格式：
    open_time | total_risk | BTC | ETH | SOL
    """

    risk_data = pd.DataFrame(
        risk_records
    )

    if risk_data.empty:
        raise ValueError(
            "指定期間沒有風險資料"
        )

    required_risk_columns = {
        "score_time",
        "total_risk",
    }

    missing_columns = (
        required_risk_columns -
        set(risk_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "風險資料缺少欄位："
            f"{sorted(missing_columns)}"
        )

    risk_data["open_time"] = (
        pd.to_datetime(
            risk_data["score_time"],
            errors="coerce",
        )
        .dt.normalize()
    )

    risk_data["total_risk"] = (
        pd.to_numeric(
            risk_data["total_risk"],
            errors="coerce",
        )
    )

    market_data = (
        risk_data[
            [
                "open_time",
                "total_risk",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    for allocation in allocations:
        asset = allocation.asset
        symbol = f"{asset}USDT"

        asset_data = price_loader(
            symbol=symbol,
            interval="1d",
            start_date=start_date.isoformat(),
        )

        if asset_data.empty:
            raise ValueError(
                f"{asset} 沒有可用的歷史價格"
            )

        required_price_columns = {
            "open_time",
            "close",
        }

        missing_price_columns = (
            required_price_columns -
            set(asset_data.columns)
        )

        if missing_price_columns:
            raise ValueError(
                f"{asset} 價格資料缺少欄位："
                f"{sorted(missing_price_columns)}"
            )

        asset_data = asset_data.copy()

        asset_data["open_time"] = (
            pd.to_datetime(
                asset_data["open_time"],
                errors="coerce",
            )
            .dt.normalize()
        )

        asset_data[asset] = (
            pd.to_numeric(
                asset_data["close"],
                errors="coerce",
            )
        )

        asset_data = asset_data[
            (
                asset_data["open_time"].dt.date
                >= start_date
            )
            &
            (
                asset_data["open_time"].dt.date
                <= end_date
            )
        ]

        asset_data = (
            asset_data[
                [
                    "open_time",
                    asset,
                ]
            ]
            .dropna()
            .drop_duplicates(
                subset=["open_time"],
                keep="last",
            )
            .sort_values("open_time")
        )

        if asset_data.empty:
            raise ValueError(
                f"{asset} 在指定日期範圍沒有價格資料"
            )

        market_data = market_data.merge(
            asset_data,
            on="open_time",
            how="left",
        )

    return market_data



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


@router.post(
    "/portfolio",
    response_model=PortfolioBacktestResponse,
)
def create_portfolio_backtest(
    request: PortfolioBacktestRequest,

    repository: MarketRepository = Depends(
        get_market_repository
    ),

    price_loader: PriceHistoryLoader = Depends(
        get_price_history_loader
    ),
) -> PortfolioBacktestResponse:
    """執行多幣種投資組合回測。"""

    try:
        risk_records = (
            repository.get_risk_history(
                symbol=request.risk_symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                limit=None,
            )
        )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="資料庫目前無法使用",
        ) from exc

    if not risk_records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定期間找不到風險資料",
        )

    try:
        market_data = (
            prepare_portfolio_market_data(
                risk_records=risk_records,
                allocations=request.allocations,
                start_date=request.start_date,
                end_date=request.end_date,
                price_loader=price_loader,
            )
        )

        allocation_map = {
            allocation.asset:
                allocation.weight
            for allocation
            in request.allocations
        }

        result = run_portfolio_backtest(
            market_data=market_data,
            allocations=allocation_map,
            daily_budget=(
                request.daily_budget
            ),
            buy_min=request.buy_min,
            buy_max=request.buy_max,
            sell_min=request.sell_min,
            sell_max=request.sell_max,
            sell_pct=request.sell_pct,
            start_date=request.start_date,
            end_date=request.end_date,
            fee_rate=request.fee_rate,
        )

    except BinanceAPIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"市場價格服務失敗：{exc}",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc

    summary = (
        PortfolioBacktestSummaryResponse(
            **result.summary
        )
    )

    equity_curve = [
        PortfolioEquityPointResponse(
            date=pd.Timestamp(
                row["Date"]
            ).to_pydatetime(),

            cash=float(row["Cash"]),

            market_value=float(
                row["Market_Value"]
            ),

            equity=float(row["Equity"]),

            contributed=float(
                row["Contributed"]
            ),

            realized_pnl=float(
                row["Realized_PnL"]
            ),

            unrealized_pnl=float(
                row["Unrealized_PnL"]
            ),

            total_fees=float(
                row["Total_Fees"]
            ),

            peak_equity=float(
                row["Peak_Equity"]
            ),

            drawdown_pct=float(
                row["Drawdown_Pct"]
            ),
        )
        for _, row
        in result.equity_curve.iterrows()
    ]

    assets = [
        PortfolioAssetResultResponse(
            asset=str(row["Asset"]),

            weight=float(row["Weight"]),

            balance=float(row["Balance"]),

            last_price=float(
                row["Last_Price"]
            ),

            market_value=float(
                row["Market_Value"]
            ),

            cost_basis=float(
                row["Cost_Basis"]
            ),

            contributed=float(
                row["Contributed"]
            ),

            realized_pnl=float(
                row["Realized_PnL"]
            ),

            unrealized_pnl=float(
                row["Unrealized_PnL"]
            ),

            profit=float(row["Profit"]),

            roi_pct=float(row["ROI_Pct"]),

            fees=float(row["Fees"]),
        )
        for _, row
        in result.asset_results.iterrows()
    ]

    trades = [
        PortfolioTradeResponse(
            date=pd.Timestamp(
                row["Date"]
            ).to_pydatetime(),

            asset=str(row["Asset"]),

            type=str(row["Type"]),

            price=float(row["Price"]),

            risk=float(row["Risk"]),

            amount=float(row["Amount"]),

            value_usdt=float(
                row["Value_USDT"]
            ),

            fee=float(row["Fee"]),
        )
        for _, row
        in result.trades.iterrows()
    ]

    return PortfolioBacktestResponse(
        summary=summary,
        equity_curve=equity_curve,
        assets=assets,
        trades=trades,
    )

