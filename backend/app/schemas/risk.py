from datetime import datetime

from pydantic import BaseModel, Field


class MarketPointRequest(BaseModel):
    """單日市場與社群資料。"""

    open_time: datetime

    close: float = Field(
        gt=0,
        description="資產收盤價，必須大於 0",
    )

    volume: float = Field(
        default=0.0,
        ge=0,
        description="成交量",
    )

    fear_greed: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Fear and Greed Index，範圍為 0 到 100",
    )


class RiskCalculationRequest(BaseModel):
    """風險計算請求。"""

    records: list[MarketPointRequest]


class RiskPointResponse(BaseModel):
    """單日風險計算結果。"""

    open_time: datetime
    close: float
    price_risk: float
    social_risk: float
    total_risk: float


class RiskCalculationResponse(BaseModel):
    """風險計算 API 回傳結果。"""

    count: int
    records: list[RiskPointResponse]