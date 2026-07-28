from datetime import date, datetime, time, timedelta
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
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        取得指定資產的歷史風險資料。

        start_date 和 end_date 都包含在查詢範圍內。
        limit 為 None 時，回傳日期範圍內的全部資料。

        最終結果按照時間由舊到新排列。
        """

        conditions = [
            "symbol = :symbol",
        ]

        parameters: dict[str, Any] = {
            "symbol": symbol,
        }

        if start_date is not None:
            start_at = datetime.combine(
                start_date,
                time.min,
            )

            conditions.append(
                "score_time >= :start_at"
            )

            parameters["start_at"] = start_at

        if end_date is not None:
            # 使用「隔天 00:00 之前」，確保 end_date 當天
            # 任何時間的資料都會包含在結果中。
            end_exclusive = datetime.combine(
                end_date + timedelta(days=1),
                time.min,
            )

            conditions.append(
                "score_time < :end_exclusive"
            )

            parameters["end_exclusive"] = (
                end_exclusive
            )

        where_clause = " AND ".join(conditions)

        limit_clause = ""

        if limit is not None:
            limit_clause = "LIMIT :limit"
            parameters["limit"] = limit

        query = text(
            f"""
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
                WHERE {where_clause}
                ORDER BY score_time DESC
                {limit_clause}
            ) AS selected_records
            ORDER BY score_time ASC
            """
        )

        with self._engine.connect() as connection:
            rows = (
                connection.execute(
                    query,
                    parameters,
                )
                .mappings()
                .all()
            )

        return [
            dict(row)
            for row in rows
        ]