import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text


@st.cache_resource
def get_engine():
    database_url = st.secrets["postgres"]["database_url"]
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )


def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT now() AS current_time"))
        return result.fetchone()
def read_youtube_metric():
    engine = get_engine()
    query = """
        SELECT
            metric_date AS date,
            video_count,
            avg_views,
            high_view_ratio,
            composite_score
        FROM youtube_metric
        ORDER BY metric_date ASC
    """
    return pd.read_sql(query, engine)


def save_youtube_metric(df):
    if df.empty:
        return 0

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "metric_date": pd.to_datetime(row["date"]).date(),
            "video_count": int(row.get("video_count", 0) or 0),
            "avg_views": float(row.get("avg_views", 0) or 0),
            "high_view_ratio": float(row.get("high_view_ratio", 0) or 0),
            "composite_score": float(row.get("composite_score", 0) or 0),
        })

    sql = text("""
        INSERT INTO youtube_metric
            (metric_date, video_count, avg_views, high_view_ratio, composite_score)
        VALUES
            (:metric_date, :video_count, :avg_views, :high_view_ratio, :composite_score)
        ON CONFLICT (metric_date)
        DO UPDATE SET
            video_count = EXCLUDED.video_count,
            avg_views = EXCLUDED.avg_views,
            high_view_ratio = EXCLUDED.high_view_ratio,
            composite_score = EXCLUDED.composite_score
    """)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)
def read_wiki_metric():
    engine = get_engine()
    query = """
        SELECT metric_date AS date_wiki, wiki_views
        FROM wiki_metric
        ORDER BY metric_date ASC
    """
    return pd.read_sql(query, engine)


def save_wiki_metric(df):
    if df.empty:
        return 0

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "metric_date": pd.to_datetime(row["date_wiki"]).date(),
            "wiki_views": int(row.get("wiki_views", 0) or 0),
        })

    sql = text("""
        INSERT INTO wiki_metric
            (metric_date, wiki_views)
        VALUES
            (:metric_date, :wiki_views)
        ON CONFLICT (metric_date)
        DO UPDATE SET
            wiki_views = EXCLUDED.wiki_views
    """)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)
def save_risk_score(df, symbol="BTCUSDT"):
    if df.empty or "total_risk" not in df.columns:
        return 0

    rows = []

    for _, row in df.iterrows():
        if pd.isna(row.get("total_risk")):
            continue

        risk = float(row["total_risk"])

        if risk <= 0.4:
            level = "BUY"
        elif risk >= 0.8:
            level = "SELL"
        else:
            level = "HODL"

        rows.append({
            "symbol": symbol,
            "score_time": pd.to_datetime(row["open_time"]).to_pydatetime(),
            "price": float(row["close"]) if not pd.isna(row.get("close")) else None,
            "total_risk": risk,
            "price_risk": float(row.get("price_risk", 0) or 0),
            "social_risk": float(row.get("social_risk", 0) or 0),
            "risk_level": level,
        })

    if not rows:
        return 0

    sql = text("""
        INSERT INTO risk_score
            (symbol, score_time, price, total_risk, price_risk, social_risk, risk_level)
        VALUES
            (:symbol, :score_time, :price, :total_risk, :price_risk, :social_risk, :risk_level)
        ON CONFLICT (symbol, score_time)
        DO UPDATE SET
            price = EXCLUDED.price,
            total_risk = EXCLUDED.total_risk,
            price_risk = EXCLUDED.price_risk,
            social_risk = EXCLUDED.social_risk,
            risk_level = EXCLUDED.risk_level
    """)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)
def save_market_price(df, symbol="BTCUSDT"):
    if df.empty:
        return 0

    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "symbol": symbol,
            "open_time": pd.to_datetime(row["open_time"]).to_pydatetime(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })

    sql = text("""
        INSERT INTO market_price
            (symbol, open_time, open, high, low, close, volume)
        VALUES
            (:symbol, :open_time, :open, :high, :low, :close, :volume)
        ON CONFLICT (symbol, open_time)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


def read_market_price(symbol="BTCUSDT"):
    engine = get_engine()
    query = text("""
        SELECT open_time, open, high, low, close, volume
        FROM market_price
        WHERE symbol = :symbol
        ORDER BY open_time ASC
    """)
    return pd.read_sql(query, engine, params={"symbol": symbol})


def write_sync_log(source_name, status, rows_inserted=0, error_message=None, latency_ms=None):
    sql = text("""
        INSERT INTO data_sync_log
            (source_name, status, rows_inserted, error_message, latency_ms)
        VALUES
            (:source_name, :status, :rows_inserted, :error_message, :latency_ms)
    """)

    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(sql, {
                "source_name": source_name,
                "status": status,
                "rows_inserted": rows_inserted,
                "error_message": error_message,
                "latency_ms": latency_ms,
            })
    except Exception as e:
        print(f"⚠️ 寫入 data_sync_log 失敗：{e}")