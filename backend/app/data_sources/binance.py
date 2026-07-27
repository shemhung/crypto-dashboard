from datetime import datetime
import time
from typing import Callable, Optional

import pandas as pd
import requests

from services.http_client import get_with_auto_proxy


class BinanceAPIError(RuntimeError):
    """Binance API 或網路連線發生錯誤。"""


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    start_date: str = "2017-08-17",
    progress_callback: Optional[Callable[[int], None]] = None,
) -> pd.DataFrame:
    """
    從 Binance 取得歷史 K 線資料。

    此函式只負責：
    1. 呼叫 Binance API
    2. 整理資料
    3. 回傳 DataFrame

    不負責顯示 Streamlit 畫面。
    """

    url = "https://api.binance.com/api/v3/klines"
    all_data = []

    start_time = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_time = int(datetime.now().timestamp() * 1000)
    current_start = start_time

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000,
        }

        try:
            response = get_with_auto_proxy(
                url,
                params=params,
                headers=headers,
                timeout=20,
            )

            if response.status_code != 200:
                raise BinanceAPIError(
                    f"Binance HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Binance K 線資料的第 7 個欄位是 close_time
            last_close_time = data[-1][6]
            current_start = last_close_time + 1

            if progress_callback is not None:
                progress_callback(len(all_data))

            # 少於 1000 筆代表已經接近最新資料
            if len(data) < 1000:
                break

            time.sleep(0.2)

        except requests.exceptions.ProxyError as exc:
            raise BinanceAPIError(
                f"Binance Proxy 連線失敗：{exc}"
            ) from exc

        except requests.exceptions.ConnectTimeout as exc:
            raise BinanceAPIError(
                f"Binance 連線逾時：{exc}"
            ) from exc

        except requests.exceptions.SSLError as exc:
            raise BinanceAPIError(
                f"Binance SSL 連線錯誤：{exc}"
            ) from exc

        except BinanceAPIError:
            raise

        except Exception as exc:
            raise BinanceAPIError(
                f"Binance 發生未知錯誤：{type(exc).__name__}: {exc}"
            ) from exc

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    if not all_data:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    df = pd.DataFrame(all_data, columns=columns)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df[
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .dropna()
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    return df