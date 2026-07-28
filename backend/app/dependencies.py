from fastapi import HTTPException, status

from backend.app.core.database import (
    DatabaseConfigurationError,
    get_engine,
)
from backend.app.repositories.market_repository import (
    MarketRepository,
)


def get_market_repository() -> MarketRepository:
    """建立提供給 API 使用的 MarketRepository。"""

    try:
        return MarketRepository(
            engine=get_engine()
        )

    except DatabaseConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc