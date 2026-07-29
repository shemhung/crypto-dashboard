from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SystemStatus = Literal[
    "healthy",
    "degraded",
    "unavailable",
]


class DatabaseStatusResponse(BaseModel):
    """資料庫連線狀態。"""

    connected: bool


class RiskDataStatusResponse(BaseModel):
    """風險歷史資料狀態。"""

    symbol: str

    record_count: int = Field(
        ge=0,
    )

    earliest_time: datetime | None
    latest_time: datetime | None

    age_hours: float | None = Field(
        default=None,
        ge=0,
    )

    is_stale: bool


class MarketApiStatusResponse(BaseModel):
    """外部市場資料 API 狀態。"""

    provider: str = "Binance"
    available: bool


class DataStatusResponse(BaseModel):
    """Data Status API 完整回應。"""

    status: SystemStatus
    checked_at: datetime

    database: DatabaseStatusResponse
    risk_data: RiskDataStatusResponse
    market_api: MarketApiStatusResponse