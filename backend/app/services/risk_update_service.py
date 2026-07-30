from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)

import pandas as pd

from backend.app.data_sources.binance import (
    fetch_binance_klines,
)
from backend.app.data_sources.fear_greed import (
    fetch_fear_greed_history,
)
from backend.app.repositories.risk_update_repository import (
    RiskUpdateRepository,
)
from backend.app.services.risk_service import (
    compute_risk,
)


PriceLoader = Callable[
    ...,
    pd.DataFrame,
]

FearGreedLoader = Callable[
    [],
    pd.DataFrame,
]


class RiskUpdateError(RuntimeError):
    """每日風險更新無法完成。"""


@dataclass(frozen=True)
class RiskUpdateResult:
    symbol: str
    market_rows: int
    risk_rows: int
    latest_score_time: datetime
    latest_price: float
    latest_total_risk: float


def _utc_day_start(
    value: datetime | None,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(
        value
        or datetime.now(timezone.utc)
    )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return (
        timestamp
        .normalize()
        .tz_localize(None)
    )


def _prepare_market_data(
    data: pd.DataFrame,
    now: datetime | None,
) -> pd.DataFrame:
    required_columns = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise RiskUpdateError(
            "Binance 資料缺少欄位："
            f"{sorted(missing_columns)}"
        )

    result = data.copy()

    result["open_time"] = (
        pd.to_datetime(
            result["open_time"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    current_day_utc = _utc_day_start(
        now
    )

    return (
        result
        .dropna(
            subset=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .loc[
            lambda frame:
            frame["open_time"]
            < current_day_utc
        ]
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def _merge_fear_greed(
    market_data: pd.DataFrame,
    fear_greed_data: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "fear_greed",
    }

    missing_columns = (
        required_columns
        - set(fear_greed_data.columns)
    )

    if missing_columns:
        raise RiskUpdateError(
            "Fear & Greed 資料缺少欄位："
            f"{sorted(missing_columns)}"
        )

    metrics = fear_greed_data.copy()

    metrics["date_only"] = (
        pd.to_datetime(
            metrics["date"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
        .dt.date
    )

    metrics["fear_greed"] = (
        pd.to_numeric(
            metrics["fear_greed"],
            errors="coerce",
        )
    )

    metrics = (
        metrics[
            [
                "date_only",
                "fear_greed",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["date_only"],
            keep="last",
        )
    )

    result = market_data.copy()

    result["date_only"] = (
        result["open_time"].dt.date
    )

    result = result.merge(
        metrics,
        on="date_only",
        how="left",
    )

    result["fear_greed"] = (
        result["fear_greed"]
        .interpolate()
        .ffill()
        .bfill()
    )

    if (
        result["fear_greed"]
        .isna()
        .all()
    ):
        raise RiskUpdateError(
            "無法將 Fear & Greed 資料"
            "合併到市場資料"
        )

    return result


def _merge_wiki(
    market_data: pd.DataFrame,
    wiki_data: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "date_wiki",
        "wiki_views",
    }

    missing_columns = (
        required_columns
        - set(wiki_data.columns)
    )

    if missing_columns:
        raise RiskUpdateError(
            "Wikipedia 資料缺少欄位："
            f"{sorted(missing_columns)}"
        )

    metrics = wiki_data.copy()

    metrics["date_only"] = (
        pd.to_datetime(
            metrics["date_wiki"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
        .dt.date
    )

    metrics["wiki_views"] = (
        pd.to_numeric(
            metrics["wiki_views"],
            errors="coerce",
        )
    )

    metrics = (
        metrics[
            [
                "date_only",
                "wiki_views",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["date_only"],
            keep="last",
        )
    )

    result = market_data.merge(
        metrics,
        on="date_only",
        how="left",
    )

    result["wiki_views"] = (
        result["wiki_views"]
        .interpolate()
        .ffill()
        .bfill()
    )

    if (
        result["wiki_views"]
        .isna()
        .all()
    ):
        raise RiskUpdateError(
            "無法將 Wikipedia 資料"
            "合併到市場資料"
        )

    return result


def _prepare_youtube_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "video_count",
        "avg_views",
        "high_view_ratio",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise RiskUpdateError(
            "YouTube 資料缺少欄位："
            f"{sorted(missing_columns)}"
        )

    result = data.copy()

    result["date"] = (
        pd.to_datetime(
            result["date"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
    )

    for column in [
        "video_count",
        "avg_views",
        "high_view_ratio",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = (
        result
        .dropna(subset=["date"])
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if result.empty:
        raise RiskUpdateError(
            "YouTube 指標資料為空"
        )

    return result


def update_daily_risk(
    repository: RiskUpdateRepository,
    symbol: str = "BTCUSDT",
    start_date: str = "2017-08-17",
    price_loader: PriceLoader = (
        fetch_binance_klines
    ),
    fear_greed_loader: FearGreedLoader = (
        fetch_fear_greed_history
    ),
    now: datetime | None = None,
) -> RiskUpdateResult:
    normalized_symbol = symbol.upper()

    market_data = price_loader(
        symbol=normalized_symbol,
        interval="1d",
        start_date=start_date,
    )

    market_data = _prepare_market_data(
        market_data,
        now=now,
    )

    if market_data.empty:
        raise RiskUpdateError(
            "Binance 沒有回傳已完成的日線資料"
        )

    fear_greed_data = (
        fear_greed_loader()
    )

    if fear_greed_data.empty:
        raise RiskUpdateError(
            "Fear & Greed 指標資料為空"
        )

    wiki_data = (
        repository.get_wiki_metrics()
    )

    if wiki_data.empty:
        raise RiskUpdateError(
            "Wikipedia 指標資料庫目前為空"
        )

    youtube_data = (
        repository.get_youtube_metrics()
    )

    if youtube_data.empty:
        raise RiskUpdateError(
            "YouTube 指標資料庫目前為空"
        )

    calculation_data = (
        _merge_fear_greed(
            market_data,
            fear_greed_data,
        )
    )

    calculation_data = _merge_wiki(
        calculation_data,
        wiki_data,
    )

    youtube_data = (
        _prepare_youtube_data(
            youtube_data
        )
    )

    calculation_data["date_index"] = (
        calculation_data[
            "open_time"
        ].dt.normalize()
    )

    calculation_data = (
        calculation_data.set_index(
            "date_index",
            drop=False,
        )
    )

    risk_data = compute_risk(
        calculation_data.copy(),
        df_youtube_activity=(
            youtube_data
        ),
    ).reset_index(drop=True)

    if (
        risk_data.empty
        or risk_data[
            "total_risk"
        ].isna().all()
    ):
        raise RiskUpdateError(
            "風險模型沒有產生有效結果"
        )

    market_rows = (
        repository.upsert_market_prices(
            market_data,
            symbol=normalized_symbol,
        )
    )

    risk_rows = (
        repository.upsert_risk_scores(
            risk_data,
            symbol=normalized_symbol,
        )
    )

    latest = (
        risk_data
        .sort_values("open_time")
        .iloc[-1]
    )

    return RiskUpdateResult(
        symbol=normalized_symbol,
        market_rows=market_rows,
        risk_rows=risk_rows,
        latest_score_time=(
            pd.Timestamp(
                latest["open_time"]
            ).to_pydatetime()
        ),
        latest_price=float(
            latest["close"]
        ),
        latest_total_risk=float(
            latest["total_risk"]
        ),
    )