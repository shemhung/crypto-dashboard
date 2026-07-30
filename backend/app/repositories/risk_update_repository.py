from __future__ import annotations

import pandas as pd

from sqlalchemy import text
from sqlalchemy.engine import Engine


class RiskUpdateRepository:
    """
    每日市場資料與風險評分更新所需的
    資料庫操作。
    """

    def __init__(
        self,
        engine: Engine,
    ) -> None:
        self._engine = engine

    def get_youtube_metrics(
        self,
    ) -> pd.DataFrame:
        query = text(
            """
            SELECT
                metric_date AS date,
                video_count,
                avg_views,
                high_view_ratio,
                composite_score
            FROM youtube_metric
            ORDER BY metric_date ASC
            """
        )

        return pd.read_sql(
            query,
            self._engine,
        )

    def get_wiki_metrics(
        self,
    ) -> pd.DataFrame:
        query = text(
            """
            SELECT
                metric_date AS date_wiki,
                wiki_views
            FROM wiki_metric
            ORDER BY metric_date ASC
            """
        )

        return pd.read_sql(
            query,
            self._engine,
        )

    def upsert_market_prices(
        self,
        data: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> int:
        if data.empty:
            return 0

        rows = []

        for record in data.to_dict(
            orient="records",
        ):
            open_time = pd.to_datetime(
                record["open_time"],
                utc=True,
            )

            rows.append(
                {
                    "symbol": symbol,
                    "open_time": (
                        open_time
                        .tz_convert(None)
                        .to_pydatetime()
                    ),
                    "open": float(
                        record["open"]
                    ),
                    "high": float(
                        record["high"]
                    ),
                    "low": float(
                        record["low"]
                    ),
                    "close": float(
                        record["close"]
                    ),
                    "volume": float(
                        record["volume"]
                    ),
                }
            )

        query = text(
            """
            INSERT INTO market_price
                (
                    symbol,
                    open_time,
                    open,
                    high,
                    low,
                    close,
                    volume
                )
            VALUES
                (
                    :symbol,
                    :open_time,
                    :open,
                    :high,
                    :low,
                    :close,
                    :volume
                )
            ON CONFLICT (symbol, open_time)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """
        )

        with self._engine.begin() as connection:
            connection.execute(
                query,
                rows,
            )

        return len(rows)

    def upsert_risk_scores(
        self,
        data: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> int:
        if (
            data.empty
            or "total_risk"
            not in data.columns
        ):
            return 0

        rows = []

        for record in data.to_dict(
            orient="records",
        ):
            total_risk = record.get(
                "total_risk"
            )

            if pd.isna(total_risk):
                continue

            risk = float(total_risk)

            if risk <= 0.4:
                risk_level = "BUY"
            elif risk >= 0.8:
                risk_level = "SELL"
            else:
                risk_level = "HODL"

            price = record.get("close")
            price_risk = record.get(
                "price_risk"
            )
            social_risk = record.get(
                "social_risk"
            )

            score_time = pd.to_datetime(
                record["open_time"],
                utc=True,
            )

            rows.append(
                {
                    "symbol": symbol,
                    "score_time": (
                        score_time
                        .tz_convert(None)
                        .to_pydatetime()
                    ),
                    "price": (
                        None
                        if pd.isna(price)
                        else float(price)
                    ),
                    "total_risk": risk,
                    "price_risk": (
                        0.0
                        if pd.isna(price_risk)
                        else float(price_risk)
                    ),
                    "social_risk": (
                        0.0
                        if pd.isna(social_risk)
                        else float(social_risk)
                    ),
                    "risk_level": risk_level,
                }
            )

        if not rows:
            return 0

        query = text(
            """
            INSERT INTO risk_score
                (
                    symbol,
                    score_time,
                    price,
                    total_risk,
                    price_risk,
                    social_risk,
                    risk_level
                )
            VALUES
                (
                    :symbol,
                    :score_time,
                    :price,
                    :total_risk,
                    :price_risk,
                    :social_risk,
                    :risk_level
                )
            ON CONFLICT (symbol, score_time)
            DO UPDATE SET
                price = EXCLUDED.price,
                total_risk = EXCLUDED.total_risk,
                price_risk = EXCLUDED.price_risk,
                social_risk = EXCLUDED.social_risk,
                risk_level = EXCLUDED.risk_level
            """
        )

        with self._engine.begin() as connection:
            connection.execute(
                query,
                rows,
            )

        return len(rows)