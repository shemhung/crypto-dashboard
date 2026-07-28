from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


class MarketRepository:
    """負責讀取市場與風險資料。"""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_latest_risk(
        self,
        symbol: str = "BTCUSDT",
    ) -> dict[str, Any] | None:
        """取得指定資產最新一筆風險資料。"""

        query = text(
            """
            SELECT
                symbol,
                score_time,
                price,
                total_risk,
                price_risk,
                social_risk,
                risk_level
            FROM risk_score
            WHERE symbol = :symbol
            ORDER BY score_time DESC
            LIMIT 1
            """
        )

        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    query,
                    {"symbol": symbol},
                )
                .mappings()
                .first()
            )

        if row is None:
            return None

        return dict(row)

    def get_risk_history(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        取得指定資產最近幾筆風險資料。

        最後按照時間由舊到新排列，
        方便前端直接畫折線圖。
        """

        query = text(
            """
            SELECT
                symbol,
                score_time,
                price,
                total_risk,
                price_risk,
                social_risk,
                risk_level
            FROM (
                SELECT
                    symbol,
                    score_time,
                    price,
                    total_risk,
                    price_risk,
                    social_risk,
                    risk_level
                FROM risk_score
                WHERE symbol = :symbol
                ORDER BY score_time DESC
                LIMIT :limit
            ) AS recent_records
            ORDER BY score_time ASC
            """
        )

        with self._engine.connect() as connection:
            rows = (
                connection.execute(
                    query,
                    {
                        "symbol": symbol,
                        "limit": limit,
                    },
                )
                .mappings()
                .all()
            )

        return [
            dict(row)
            for row in rows
        ]