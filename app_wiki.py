# -*- coding: utf-8 -*-
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text


DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL environment variable")


engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def get_latest_wiki_date():
    sql = text("SELECT MAX(metric_date) FROM wiki_metric")

    with engine.connect() as conn:
        result = conn.execute(sql).scalar()

    return result


def fetch_wiki_range(start_date, end_date):
    headers = {
        "User-Agent": "BitcoinRiskBot/1.0 (Personal Education Project)"
    }

    all_rows = []
    fetch_ptr = start_date

    while fetch_ptr <= end_date:
        chunk_end = min(fetch_ptr + timedelta(days=365), end_date)

        start_str = fetch_ptr.strftime("%Y%m%d")
        end_str = chunk_end.strftime("%Y%m%d")

        print(f"Fetching Wikipedia: {start_str} - {end_str}")

        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/all-agents/Bitcoin/daily/{start_str}/{end_str}"
        )

        resp = requests.get(url, headers=headers, timeout=20)

        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("items", []):
                raw_date = item["timestamp"]
                metric_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

                all_rows.append({
                    "metric_date": metric_date,
                    "wiki_views": int(item["views"])
                })
        else:
            print(f"HTTP {resp.status_code}: {resp.text[:200]}")

        fetch_ptr = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    return pd.DataFrame(all_rows)


def save_wiki_metric(df):
    if df.empty:
        return 0

    rows = []

    for _, row in df.iterrows():
        rows.append({
            "metric_date": pd.to_datetime(row["metric_date"]).date(),
            "wiki_views": int(row["wiki_views"]),
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

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


def write_sync_log(source_name, status, rows_inserted=0, error_message=None):
    sql = text("""
        INSERT INTO data_sync_log
            (source_name, status, rows_inserted, error_message)
        VALUES
            (:source_name, :status, :rows_inserted, :error_message)
    """)

    try:
        with engine.begin() as conn:
            conn.execute(sql, {
                "source_name": source_name,
                "status": status,
                "rows_inserted": rows_inserted,
                "error_message": error_message,
            })
    except Exception as e:
        print(f"Failed to write data_sync_log: {e}")


def main():
    try:
        latest_date = get_latest_wiki_date()

        if latest_date:
            start_date = pd.to_datetime(latest_date).to_pydatetime() + timedelta(days=1)
        else:
            start_date = datetime(2015, 7, 1)

        end_date = datetime.utcnow()

        if start_date.date() > end_date.date():
            print("Wikipedia data is already up to date.")
            write_sync_log("wikipedia", "SUCCESS", rows_inserted=0)
            return

        df = fetch_wiki_range(start_date, end_date)

        rows = save_wiki_metric(df)
        write_sync_log("wikipedia", "SUCCESS", rows_inserted=rows)

        print(f"Saved {rows} Wikipedia rows to Supabase.")

    except Exception as e:
        write_sync_log("wikipedia", "FAILED", error_message=str(e))
        raise


if __name__ == "__main__":
    main()