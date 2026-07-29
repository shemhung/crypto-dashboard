from fastapi import APIRouter
from datetime import (
    date,
    datetime,
    timezone,
)

from fastapi import Depends
from sqlalchemy.exc import SQLAlchemyError

from backend.app.dependencies import (
    MarketApiChecker,
    get_market_api_checker,
    get_market_repository,
)

from backend.app.repositories.market_repository import (
    MarketRepository,
)

from backend.app.schemas.health import (
    DatabaseStatusResponse,
    DataStatusResponse,
    MarketApiStatusResponse,
    RiskDataStatusResponse,
)

from backend.app.services.data_status_service import (
    summarize_risk_data,
)

router = APIRouter(
    tags=["health"],
)


@router.get("/health")
def health_check() -> dict[str, str]:
    """確認後端服務是否正常運作。"""

    return {
        "status": "ok",
        "service": "crypto-dashboard-api",
    }

@router.get(
    "/data-status",
    response_model=DataStatusResponse,
)
def get_data_status(
    repository: MarketRepository = Depends(
        get_market_repository
    ),

    market_api_checker: MarketApiChecker = Depends(
        get_market_api_checker
    ),
) -> DataStatusResponse:
    """
    取得資料庫、風險資料與市場 API 狀態。
    """

    symbol = "BTCUSDT"

    checked_at = datetime.now(
        timezone.utc
    )

    market_api_available = (
        market_api_checker()
    )

    try:
        records = (
            repository.get_risk_history(
                symbol=symbol,

                # 涵蓋 BTC 可取得的完整歷史範圍
                start_date=date(
                    2010,
                    1,
                    1,
                ),

                end_date=checked_at.date(),

                limit=None,
            )
        )

    except SQLAlchemyError:
        return DataStatusResponse(
            status="unavailable",

            checked_at=checked_at,

            database=DatabaseStatusResponse(
                connected=False,
            ),

            risk_data=RiskDataStatusResponse(
                symbol=symbol,
                record_count=0,
                earliest_time=None,
                latest_time=None,
                age_hours=None,
                is_stale=True,
            ),

            market_api=MarketApiStatusResponse(
                provider="Binance",
                available=(
                    market_api_available
                ),
            ),
        )

    risk_summary = summarize_risk_data(
        records=records,
        symbol=symbol,
        now=checked_at,
    )

    system_status = "healthy"

    if (
        risk_summary.is_stale or
        not market_api_available
    ):
        system_status = "degraded"

    return DataStatusResponse(
        status=system_status,

        checked_at=checked_at,

        database=DatabaseStatusResponse(
            connected=True,
        ),

        risk_data=RiskDataStatusResponse(
            symbol=risk_summary.symbol,

            record_count=(
                risk_summary.record_count
            ),

            earliest_time=(
                risk_summary.earliest_time
            ),

            latest_time=(
                risk_summary.latest_time
            ),

            age_hours=(
                risk_summary.age_hours
            ),

            is_stale=(
                risk_summary.is_stale
            ),
        ),

        market_api=MarketApiStatusResponse(
            provider="Binance",
            available=market_api_available,
        ),
    )