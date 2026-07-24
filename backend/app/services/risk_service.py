import math
from datetime import datetime

import numpy as np
import pandas as pd

from backend.app.core.config import DEFAULT_RISK_WEIGHTS


# 暫時保留原本的變數名稱，降低第一次搬移出錯的可能性
SOCIAL_RISK_WEIGHT = DEFAULT_RISK_WEIGHTS.social
PRICE_RISK_WEIGHT = DEFAULT_RISK_WEIGHTS.price
DERIVATIVE_RISK_WEIGHT = DEFAULT_RISK_WEIGHTS.derivative
VOLUME_RISK_WEIGHT = DEFAULT_RISK_WEIGHTS.volume

FEAR_GREED_WEIGHT = DEFAULT_RISK_WEIGHTS.fear_greed
YOUTUBE_WEIGHT = DEFAULT_RISK_WEIGHTS.youtube
WIKIPEDIA_WEIGHT = DEFAULT_RISK_WEIGHTS.wikipedia
BLOCKCHAIN_COM_WEIGHT = DEFAULT_RISK_WEIGHTS.blockchain
COINGLASS_WEIGHT = DEFAULT_RISK_WEIGHTS.coinglass
GOOGLE_NEWS_WEIGHT = DEFAULT_RISK_WEIGHTS.google_news
BTC_OBITUARIES_WEIGHT = DEFAULT_RISK_WEIGHTS.obituaries
CMC_TRENDING_WEIGHT = DEFAULT_RISK_WEIGHTS.cmc_trending

def compute_rainbow_risk(current_price):
    try:
        genesis_date = datetime(2009, 1, 3)
        days_since_genesis = (datetime.now() - genesis_date).days
        if days_since_genesis > 0:
            log_days = math.log10(days_since_genesis)
            upper_band = 10 ** (5.84 * log_days - 17.01)
            lower_band = 10 ** (5.84 * log_days - 18.0)
            if current_price >= upper_band: return 1.0
            elif current_price <= lower_band: return 0.0
            else: return (current_price - lower_band) / (upper_band - lower_band)
    except: pass
    return 0.5

def compute_risk(df, df_blockchain_com=None, df_coinglass=None, df_obituaries=None, df_google_news=None, cmc_rank=None, df_youtube_activity=None):
    
    price_max = df["close"].max()
    price_min = df["close"].min()
    price_range = price_max - price_min
    price_risk_basic = (df["close"] - price_min) / price_range if price_range > 0 else 0.5
    rainbow_risk = df['close'].apply(compute_rainbow_risk)
    price_risk = 0.5 * price_risk_basic + 0.5 * rainbow_risk
    price_risk = np.sqrt(price_risk.clip(0, 1))

    social_risk_components = []
    
    if 'social_interest' not in df.columns:
        df['social_interest'] = 50 

    if 'fear_greed' in df.columns and not df['fear_greed'].isna().all():
        fg_risk = df['fear_greed'].rank(pct=True).rolling(7, min_periods=1).mean().clip(0, 1)
        social_risk_components.append(('fear_greed', fg_risk, FEAR_GREED_WEIGHT))

    if df_youtube_activity is not None and not df_youtube_activity.empty:
        df_youtube_activity = df_youtube_activity.copy()

        # 日期統一成 datetime64[ns]
        df_youtube_activity["date"] = pd.to_datetime(
            df_youtube_activity["date"],
            errors="coerce"
        ).dt.tz_localize(None).astype("datetime64[ns]")

        df_youtube_activity = (
            df_youtube_activity
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        # 數值欄位防呆
        for col in ["video_count", "avg_views", "high_view_ratio"]:
            if col not in df_youtube_activity.columns:
                df_youtube_activity[col] = 0
            df_youtube_activity[col] = pd.to_numeric(
                df_youtube_activity[col],
                errors="coerce"
            ).fillna(0)

        videos_norm = (
            (df_youtube_activity["video_count"] - df_youtube_activity["video_count"].min()) /
            (df_youtube_activity["video_count"].max() - df_youtube_activity["video_count"].min() + 1)
        )

        log_views = np.log10(df_youtube_activity["avg_views"] + 1)
        views_norm = (
            (log_views - log_views.min()) /
            (log_views.max() - log_views.min() + 0.001)
        )

        heat_norm = df_youtube_activity["high_view_ratio"]

        df_youtube_activity["composite_score"] = (
            0.3 * videos_norm +
            0.4 * views_norm +
            0.3 * heat_norm
        )

        temp_df = df[["open_time"]].copy()

        temp_df["date"] = pd.to_datetime(
            temp_df["open_time"],
            errors="coerce"
        ).dt.tz_localize(None).astype("datetime64[ns]")

        temp_df = (
            temp_df
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        yt_for_merge = (
            df_youtube_activity[["date", "composite_score"]]
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        merged_yt = pd.merge_asof(
            temp_df,
            yt_for_merge,
            on="date",
            direction="backward"
        )

        df["youtube_val"] = merged_yt["composite_score"].values
        df["youtube_val"] = df["youtube_val"].ffill().bfill().fillna(0)

        yt_risk = (
            df["youtube_val"]
            .rank(pct=True)
            .rolling(7, min_periods=1)
            .mean()
            .clip(0, 1)
        )

        social_risk_components.append(("youtube", yt_risk, YOUTUBE_WEIGHT))

    else:
        df["youtube_val"] = 0

    if 'wiki_views' in df.columns and not df['wiki_views'].isna().all():
        wiki_val = df['wiki_views'].copy()
        wiki_risk = wiki_val.rank(pct=True).rolling(7, min_periods=1).mean().clip(0, 1)
        social_risk_components.append(('wiki', wiki_risk, WIKIPEDIA_WEIGHT))
        
    if df_blockchain_com is not None and not df_blockchain_com.empty:
        df_blockchain_com = df_blockchain_com.set_index('date_blockchain')
        df['blockchain_active'] = df.index.map(lambda x: df_blockchain_com['unique_addresses'].reindex([x], method='nearest').iloc[0] if not df_blockchain_com.empty else np.nan)
        df['blockchain_active'] = df['blockchain_active'].ffill().bfill()
        bc_risk = df['blockchain_active'].rank(pct=True).rolling(7, min_periods=1).mean().clip(0, 1)
        social_risk_components.append(('blockchain', bc_risk, BLOCKCHAIN_COM_WEIGHT))

    if df_coinglass is not None and not df_coinglass.empty:
        df_cg = df_coinglass.groupby(df_coinglass['date_coinglass'].dt.date)['funding_rate'].mean()
        df['cg_funding'] = df.index.map(lambda x: df_cg.get(x.date(), np.nan))
        df['cg_funding'] = df['cg_funding'].ffill().bfill().fillna(0)
        cg_risk = ((df['cg_funding'] + 0.001) / 0.003).clip(0, 1).rank(pct=True).rolling(7, min_periods=1).mean()
        social_risk_components.append(('coinglass', cg_risk, COINGLASS_WEIGHT))
    
    if df_google_news is not None and not df_google_news.empty:
        df_gn = df_google_news.set_index('date_news')
        df['news_count'] = df.index.map(lambda x: df_gn['news_count'].reindex([x], method='nearest').iloc[0] if not df_gn.empty else np.nan)
        df['news_count'] = df['news_count'].ffill().bfill()
        news_risk = df['news_count'].rank(pct=True).rolling(7, min_periods=1).mean().clip(0, 1)
        social_risk_components.append(('news', news_risk, GOOGLE_NEWS_WEIGHT))

    if df_obituaries is not None and not df_obituaries.empty:
        obt_risk = pd.Series([0.2]*len(df), index=df.index)
        social_risk_components.append(('obituaries', obt_risk, BTC_OBITUARIES_WEIGHT))

    if cmc_rank is not None:
        cmc_risk_val = 1 - (cmc_rank / 100)
        cmc_risk = pd.Series([cmc_risk_val] * len(df), index=df.index)
        df['cmc_rank'] = cmc_rank
        social_risk_components.append(('cmc', cmc_risk, CMC_TRENDING_WEIGHT))

    if social_risk_components:
        total_w = sum([w for _,_,w in social_risk_components])
        social_risk = pd.Series(0.0, index=df.index)
        if total_w > 0:
            for _, risk_s, w in social_risk_components:
                social_risk += risk_s * (w / total_w)
    else:
        social_risk = pd.Series(0.5, index=df.index)

    if not social_risk.isna().all():
        total_risk = SOCIAL_RISK_WEIGHT * social_risk + PRICE_RISK_WEIGHT * price_risk + VOLUME_RISK_WEIGHT * 0 
    else:
        total_risk = 0.8 * price_risk + 0.2 * 0.5

    df["price_risk"] = price_risk
    df["social_risk"] = social_risk
    df["total_risk"] = total_risk
    
    return df

