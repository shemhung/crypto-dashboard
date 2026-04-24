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

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, {
            "source_name": source_name,
            "status": status,
            "rows_inserted": rows_inserted,
            "error_message": error_message,
            "latency_ms": latency_ms,
        })