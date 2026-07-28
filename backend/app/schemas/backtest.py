from datetime import date, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)


class BacktestMarketPoint(BaseModel):
    """單日回測資料。"""

    open_time: datetime

    asset_price: float = Field(
        gt=0,
        description="資產價格，必須大於 0",
    )

    total_risk: float = Field(
        ge=0,
        le=1,
        description="風險值，範圍為 0 到 1",
    )


class BacktestRequest(BaseModel):
    """單一資產回測請求。"""

    records: list[BacktestMarketPoint] = Field(
        min_length=1,
    )

    buy_amount: float = Field(
        gt=0,
        description="每次買入金額",
    )

    buy_min: float = Field(
        ge=0,
        le=1,
    )

    buy_max: float = Field(
        ge=0,
        le=1,
    )

    sell_pct: float = Field(
        gt=0,
        le=1,
        description="每次賣出比例，例如 0.5 代表 50%",
    )

    sell_min: float = Field(
        ge=0,
        le=1,
    )

    sell_max: float = Field(
        ge=0,
        le=1,
    )

    start_date: date

    fee_rate: float = Field(
        default=0.001,
        ge=0,
        le=0.1,
        description="交易手續費率，例如 0.001 代表 0.1%",
    )

    @model_validator(mode="after")
    def validate_risk_ranges(
        self,
    ) -> "BacktestRequest":
        """確認買賣區間設定合理。"""

        if self.buy_min >= self.buy_max:
            raise ValueError(
                "buy_min 必須小於 buy_max"
            )

        if self.sell_min >= self.sell_max:
            raise ValueError(
                "sell_min 必須小於 sell_max"
            )

        return self


class TradeResponse(BaseModel):
    """單筆交易紀錄。"""

    date: datetime
    type: Literal["BUY", "SELL"]
    price: float
    risk: float
    value_usdt: float
    amount: float
    fee: float
    balance: float


class PortfolioPointResponse(BaseModel):
    """單日資產組合狀態。"""

    date: datetime
    equity: float
    invested: float
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    avg_cost: float
    peak_equity: float


class BacktestResponse(BaseModel):
    """回測結果。"""

    trade_count: int
    buy_days: int
    sell_days: int
    final_price: float
    trades: list[TradeResponse]
    portfolio: list[PortfolioPointResponse]