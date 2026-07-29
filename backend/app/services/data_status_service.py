from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RiskDataSummary:
    """風險歷史資料摘要。"""

    symbol: str
    record_count: int
    earliest_time: datetime | None
    latest_time: datetime | None
    age_hours: float | None
    is_stale: bool


def normalize_utc_datetime(
    value: object,
) -> datetime | None:
    """
    將日期統一轉成 UTC aware datetime。
    """

    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(timestamp):
        return None

    return pd.Timestamp(
        timestamp
    ).to_pydatetime()


def get_record_score_time(
    record: object,
) -> object | None:
    """從 dictionary 或物件讀取 score_time。"""

    if isinstance(record, Mapping):
        return record.get("score_time")

    return getattr(
        record,
        "score_time",
        None,
    )


def summarize_risk_data(
    records: Iterable[
        Mapping[str, Any] | object
    ],
    symbol: str,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(
        hours=48,
    ),
) -> RiskDataSummary:
    """計算風險歷史資料的新鮮度。"""

    normalized_symbol = (
        symbol.strip().upper()
    )

    if not normalized_symbol:
        raise ValueError(
            "symbol 不可為空"
        )

    if stale_after.total_seconds() <= 0:
        raise ValueError(
            "stale_after 必須大於 0"
        )

    checked_at = normalize_utc_datetime(
        now or datetime.now(
            timezone.utc
        )
    )

    if checked_at is None:
        raise ValueError(
            "now 不是有效日期"
        )

    valid_times: list[datetime] = []

    for record in records:
        score_time = get_record_score_time(
            record
        )

        normalized_time = (
            normalize_utc_datetime(
                score_time
            )
        )

        if normalized_time is not None:
            valid_times.append(
                normalized_time
            )

    if not valid_times:
        return RiskDataSummary(
            symbol=normalized_symbol,
            record_count=0,
            earliest_time=None,
            latest_time=None,
            age_hours=None,
            is_stale=True,
        )

    earliest_time = min(valid_times)
    latest_time = max(valid_times)

    age_seconds = max(
        0.0,
        (
            checked_at -
            latest_time
        ).total_seconds(),
    )

    age_hours = age_seconds / 3600

    is_stale = (
        checked_at - latest_time
        > stale_after
    )

    return RiskDataSummary(
        symbol=normalized_symbol,
        record_count=len(valid_times),
        earliest_time=earliest_time,
        latest_time=latest_time,
        age_hours=age_hours,
        is_stale=is_stale,
    )