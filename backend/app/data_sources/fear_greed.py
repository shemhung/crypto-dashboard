from __future__ import annotations

import pandas as pd
import requests


class FearGreedAPIError(RuntimeError):
    """Fear & Greed API 或資料格式發生錯誤。"""


def fetch_fear_greed_history() -> pd.DataFrame:
    """取得完整 Fear & Greed 歷史資料。"""

    url = "https://api.alternative.me/fng/"

    try:
        response = requests.get(
            url,
            params={
                "limit": 0,
                "format": "json",
            },
            timeout=20,
        )

        response.raise_for_status()
        payload = response.json()

    except (
        requests.RequestException,
        ValueError,
    ) as exc:
        raise FearGreedAPIError(
            f"Fear & Greed API 連線失敗：{exc}"
        ) from exc

    records = payload.get("data")

    if (
        not isinstance(records, list)
        or not records
    ):
        raise FearGreedAPIError(
            "Fear & Greed API 沒有回傳可用資料"
        )

    data = pd.DataFrame(records)

    required_columns = {
        "timestamp",
        "value",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise FearGreedAPIError(
            "Fear & Greed 資料缺少欄位："
            f"{sorted(missing_columns)}"
        )

    data["date"] = (
        pd.to_datetime(
            pd.to_numeric(
                data["timestamp"],
                errors="coerce",
            ),
            unit="s",
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    data["fear_greed"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    return (
        data[
            [
                "date",
                "fear_greed",
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )