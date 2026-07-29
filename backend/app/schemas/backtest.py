from datetime import date, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
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

class PortfolioAllocationRequest(BaseModel):
    """單一幣種及其投資權重。"""

    asset: str = Field(
        min_length=2,
        max_length=15,
        pattern=r"^[A-Za-z0-9]+$",
        description="幣種代號，例如 BTC、ETH、SOL",
    )

    weight: float = Field(
        gt=0,
        le=1,
        description="配置權重，例如 0.6 代表 60%",
    )

    @field_validator("asset")
    @classmethod
    def normalize_asset(
        cls,
        value: str,
    ) -> str:
        return value.upper()


class PortfolioBacktestRequest(BaseModel):
    """投資組合回測請求。"""

    risk_symbol: str = Field(
        default="BTCUSDT",
        min_length=3,
        max_length=20,
        pattern=r"^[A-Za-z0-9]+$",
    )

    start_date: date
    end_date: date

    daily_budget: float = Field(
        gt=0,
        description="觸發買入條件時，每日投入總金額",
    )

    buy_min: float = Field(
        ge=0,
        le=1,
    )

    buy_max: float = Field(
        ge=0,
        le=1,
    )

    sell_min: float = Field(
        ge=0,
        le=1,
    )

    sell_max: float = Field(
        ge=0,
        le=1,
    )

    sell_pct: float = Field(
        gt=0,
        le=1,
        description="每次賣出的持倉比例",
    )

    fee_rate: float = Field(
        default=0.001,
        ge=0,
        le=0.1,
    )

    allocations: list[
        PortfolioAllocationRequest
    ] = Field(
        min_length=1,
    )

    @field_validator("risk_symbol")
    @classmethod
    def normalize_risk_symbol(
        cls,
        value: str,
    ) -> str:
        return value.upper()

    @model_validator(mode="after")
    def validate_request(
        self,
    ) -> "PortfolioBacktestRequest":
        if self.start_date > self.end_date:
            raise ValueError(
                "start_date 不可晚於 end_date"
            )

        if self.buy_min >= self.buy_max:
            raise ValueError(
                "buy_min 必須小於 buy_max"
            )

        if self.sell_min >= self.sell_max:
            raise ValueError(
                "sell_min 必須小於 sell_max"
            )

        if self.buy_max > self.sell_min:
            raise ValueError(
                "買入與賣出風險區間不可重疊"
            )

        assets = [
            allocation.asset
            for allocation in self.allocations
        ]

        if len(assets) != len(set(assets)):
            raise ValueError(
                "同一幣種不可重複配置"
            )

        weight_total = sum(
            allocation.weight
            for allocation in self.allocations
        )

        if abs(weight_total - 1.0) > 1e-6:
            raise ValueError(
                "幣種權重總和必須等於 1.0"
            )

        return self


class PortfolioBacktestSummaryResponse(BaseModel):
    """投資組合核心績效指標。"""

    total_contributed: float
    total_equity: float
    total_profit: float
    roi_pct: float
    mdd_pct: float

    buy_days: int
    sell_days: int
    trade_count: int

    cash_balance: float
    market_value: float

    total_fees: float
    realized_pnl: float
    unrealized_pnl: float


class PortfolioEquityPointResponse(BaseModel):
    """單日投資組合淨值。"""

    date: datetime
    cash: float
    market_value: float
    equity: float
    contributed: float
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    peak_equity: float
    drawdown_pct: float


class PortfolioAssetResultResponse(BaseModel):
    """單一幣種回測結果。"""

    asset: str
    weight: float
    balance: float
    last_price: float
    market_value: float
    cost_basis: float
    contributed: float
    realized_pnl: float
    unrealized_pnl: float
    profit: float
    roi_pct: float
    fees: float


class PortfolioTradeResponse(BaseModel):
    """投資組合交易紀錄。"""

    date: datetime
    asset: str
    type: Literal["BUY", "SELL"]
    price: float
    risk: float
    amount: float
    value_usdt: float
    fee: float


class PortfolioBacktestResponse(BaseModel):
    """投資組合回測完整結果。"""

    summary: PortfolioBacktestSummaryResponse

    equity_curve: list[
        PortfolioEquityPointResponse
    ]

    assets: list[
        PortfolioAssetResultResponse
    ]

    trades: list[
        PortfolioTradeResponse
    ]
    