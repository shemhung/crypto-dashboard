# -*- coding: utf-8 -*-
import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text


DATABASE_URL = os.environ.get("DATABASE_URL")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL environment variable")

if not YOUTUBE_API_KEY:
    raise RuntimeError("Missing YOUTUBE_API_KEY environment variable")


engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def fetch_youtube_monthly_data(year, month, keywords=("Bitcoin", "BTC")):
    start_date = datetime(year, month, 1)

    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

    print(f"Fetching YouTube: {year}-{month:02d}")

    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "key": YOUTUBE_API_KEY,
        "q": " OR ".join(keywords),
        "part": "snippet",
        "type": "video",
        "maxResults": 50,
        "publishedAfter": start_str,
        "publishedBefore": end_str,
        "order": "relevance",
    }

    resp = requests.get(search_url, params=search_params, timeout=20)

    if resp.status_code != 200:
        raise RuntimeError(f"YouTube search API error {resp.status_code}: {resp.text[:500]}")

    videos = resp.json().get("items", [])
    video_count = len(videos)

    if not videos:
        return None

    video_ids = [
        v["id"]["videoId"]
        for v in videos
        if "id" in v and "videoId" in v["id"]
    ]

    stats_url = "https://www.googleapis.com/youtube/v3/videos"
    stats_params = {
        "key": YOUTUBE_API_KEY,
        "id": ",".join(video_ids),
        "part": "statistics",
    }

    stats_resp = requests.get(stats_url, params=stats_params, timeout=20)

    views_list = []

    if stats_resp.status_code == 200:
        for item in stats_resp.json().get("items", []):
            views = int(item.get("statistics", {}).get("viewCount", 0))
            views_list.append(views)
    else:
        raise RuntimeError(f"YouTube videos API error {stats_resp.status_code}: {stats_resp.text[:500]}")

    total_views = sum(views_list)
    avg_views = float(np.mean(views_list)) if views_list else 0.0
    high_view_ratio = (
        sum(1 for v in views_list if v > 100000) / len(views_list)
        if views_list else 0.0
    )

    return {
        "metric_date": start_date.date(),
        "video_count": int(video_count),
        "total_views": int(total_views),
        "avg_views": avg_views,
        "high_view_ratio": float(high_view_ratio),
    }


def save_youtube_metric(rows):
    if not rows:
        return 0

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

    prepared = []

    for row in rows:
        video_count_norm = min(row["video_count"] / 50, 1)
        avg_views_norm = min(np.log10(row["avg_views"] + 1) / 6, 1)
        high_view_ratio = row["high_view_ratio"]

        composite_score = (
            0.3 * video_count_norm +
            0.4 * avg_views_norm +
            0.3 * high_view_ratio
        )

        prepared.append({
            "metric_date": row["metric_date"],
            "video_count": row["video_count"],
            "avg_views": row["avg_views"],
            "high_view_ratio": row["high_view_ratio"],
            "composite_score": float(composite_score),
        })

    with engine.begin() as conn:
        conn.execute(sql, prepared)

    return len(prepared)


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


def main(months_back=1):
    try:
        today = datetime.utcnow()
        rows = []

        for i in range(months_back):
            d = today - timedelta(days=i * 30)
            result = fetch_youtube_monthly_data(d.year, d.month)

            if result:
                rows.append(result)

            time.sleep(1)

        saved = save_youtube_metric(rows)
        write_sync_log("youtube", "SUCCESS", rows_inserted=saved)

        print(f"Saved {saved} YouTube rows to Supabase.")

    except Exception as e:
        write_sync_log("youtube", "FAILED", error_message=str(e))
        raise


if __name__ == "__main__":
    months = 1

    if len(sys.argv) > 1:
        months = int(sys.argv[1])

    main(months)