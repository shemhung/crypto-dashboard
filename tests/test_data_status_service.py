from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from backend.app.services.data_status_service import (
    normalize_utc_datetime,
    summarize_risk_data,
)


def test_summarize_fresh_risk_data():
    records = [
        {
            "score_time":
                "2026-07-27T00:00:00Z",
        },
        {
            "score_time":
                "2026-07-28T00:00:00+00:00",
        },
        {
            # 無時區字串也會統一成 UTC
            "score_time":
                "2026-07-29T00:00:00",
        },
    ]

    result = summarize_risk_data(
        records=records,
        symbol="btcusdt",
        now=datetime(
            2026,
            7,
            29,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert result.symbol == "BTCUSDT"
    assert result.record_count == 3

    assert result.earliest_time == datetime(
        2026,
        7,
        27,
        tzinfo=timezone.utc,
    )

    assert result.latest_time == datetime(
        2026,
        7,
        29,
        tzinfo=timezone.utc,
    )

    assert result.age_hours == pytest.approx(
        12.0
    )

    assert result.is_stale is False


def test_empty_records_are_stale():
    result = summarize_risk_data(
        records=[],
        symbol="BTCUSDT",
        now=datetime(
            2026,
            7,
            29,
            tzinfo=timezone.utc,
        ),
    )

    assert result.record_count == 0
    assert result.earliest_time is None
    assert result.latest_time is None
    assert result.age_hours is None
    assert result.is_stale is True


def test_invalid_dates_are_ignored():
    records = [
        {
            "score_time":
                "invalid-date",
        },
        {
            "score_time": None,
        },
        {
            "score_time":
                "2026-07-29T00:00:00Z",
        },
    ]

    result = summarize_risk_data(
        records=records,
        symbol="BTCUSDT",
        now=datetime(
            2026,
            7,
            29,
            12,
            tzinfo=timezone.utc,
        ),
    )

    assert result.record_count == 1

    assert result.latest_time == datetime(
        2026,
        7,
        29,
        tzinfo=timezone.utc,
    )


def test_old_data_is_stale():
    records = [
        {
            "score_time":
                "2026-07-25T00:00:00Z",
        },
    ]

    result = summarize_risk_data(
        records=records,
        symbol="BTCUSDT",
        now=datetime(
            2026,
            7,
            29,
            tzinfo=timezone.utc,
        ),
        stale_after=timedelta(
            hours=48,
        ),
    )

    assert result.age_hours == pytest.approx(
        96.0
    )

    assert result.is_stale is True


def test_future_time_does_not_create_negative_age():
    records = [
        {
            "score_time":
                "2026-07-30T00:00:00Z",
        },
    ]

    result = summarize_risk_data(
        records=records,
        symbol="BTCUSDT",
        now=datetime(
            2026,
            7,
            29,
            tzinfo=timezone.utc,
        ),
    )

    assert result.age_hours == pytest.approx(
        0.0
    )

    assert result.is_stale is False


def test_normalize_naive_datetime_to_utc():
    value = datetime(
        2026,
        7,
        29,
        12,
        30,
    )

    result = normalize_utc_datetime(
        value
    )

    assert result == datetime(
        2026,
        7,
        29,
        12,
        30,
        tzinfo=timezone.utc,
    )


def test_empty_symbol_is_rejected():
    with pytest.raises(
        ValueError,
        match="symbol 不可為空",
    ):
        summarize_risk_data(
            records=[],
            symbol="  ",
        )