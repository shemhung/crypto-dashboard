from fastapi import HTTPException, status
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import pandas as pd

from backend.app.data_sources.binance import (
    fetch_binance_klines,
)
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


PriceHistoryLoader = Callable[
    ...,
    pd.DataFrame,
]


def get_price_history_loader() -> PriceHistoryLoader:
    """
    提供歷史價格載入函式。

    正式環境使用 Binance；
    測試時可以使用 dependency override 替換。
    """

    return fetch_binance_klines


MarketApiChecker = Callable[
    [],
    bool,
]


def check_binance_api() -> bool:
    """
    檢查 Binance API 是否可以連線。

    使用 Binance 官方 ping endpoint，
    不會下載大量市場資料。
    """

    try:
        with urlopen(
            "https://api.binance.com/api/v3/ping",
            timeout=5,
        ) as response:
            return (
                200
                <= response.status
                < 300
            )

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
    ):
        return False


def get_market_api_checker(
) -> MarketApiChecker:
    """
    提供市場 API 狀態檢查函式。

    測試時可以透過 FastAPI
    dependency_overrides 換成假函式。
    """

    return check_binance_api