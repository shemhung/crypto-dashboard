# -*- coding: utf-8 -*-
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import sys
import io
import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import math
import json

# 引入回填工具
try:
    from youtube_backfill import backfill_youtube_history
except ImportError:
    pass 

try:
    from wiki_backfill import fetch_wiki_history
except ImportError:
    pass

# ============================================================
# 0. Streamlit 頁面設定
# ============================================================
st.set_page_config(
    page_title="BTC Cycle Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 優化
st.markdown("""
    <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 1. 全局設定與權重
# ============================================================

CACHE_DIR = 'cache_data'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

TRENDS_CACHE_FILE = os.path.join(CACHE_DIR, 'google_trends_cache.csv')
WIKI_CACHE_FILE = os.path.join(CACHE_DIR, 'wikipedia_views_cache.csv')
YOUTUBE_CACHE_FILE = os.path.join(CACHE_DIR, 'youtube_activity_cache.csv')

try:
    SERPAPI_KEY = st.secrets["general"]["serpapi_key"]
    YOUTUBE_API_KEY = st.secrets["general"]["youtube_api_key"]
except Exception:
    SERPAPI_KEY = ""
    YOUTUBE_API_KEY = ""

# 前 100 大主流老幣種
BACKUP_COINS = [ 
    "BNB", "TRX", "ADA", "ALGO", "ATOM", "DASH", "XTZ", "IOTA","XRP", "SOL", "BCH",  "XLM", "AAVE" , "ETC", "FIL", "QNT"
]
TOP_COINS = [
    "BTC", "ETH", "LINK", "FET", "RENDER", "DOGE", "LTC", 
]

# 【總風險權重分配】
SOCIAL_RISK_WEIGHT = 1.0      
PRICE_RISK_WEIGHT = 0.0       
DERIVATIVE_RISK_WEIGHT = 0.0  
VOLUME_RISK_WEIGHT = 0.0      

# 【社交風險內部權重】
FEAR_GREED_WEIGHT = 0.5        
YOUTUBE_WEIGHT = 0.3           
WIKIPEDIA_WEIGHT = 0.2         
BLOCKCHAIN_COM_WEIGHT = 0.0   
COINGLASS_WEIGHT = 0.0        
GOOGLE_NEWS_WEIGHT = 0.0      
BTC_OBITUARIES_WEIGHT = 0.0   
CMC_TRENDING_WEIGHT = 0.0     

# 權重為 0 的項目
TRENDS_WEIGHT = 0.00           
REDDIT_WEIGHT = 0.00           
TWITTER_WEIGHT = 0.0           
CRYPTOPANIC_WEIGHT = 0.00      
BITINFOCHARTS_WEIGHT = 0.00    
LUNARCRUSH_WEIGHT = 0.00       

# ============================================================
# 2. 數據抓取函數
# ============================================================

import requests
import pandas as pd
from datetime import datetime
import time
import streamlit as st # 記得引入 streamlit

# 移除 get_free_proxies 函數，因為我們直接用付費/穩定的 Proxy，不用爬蟲

def fetch_binance_klines(symbol="BTCUSDT", interval="1d", start_date="2017-08-17"):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    start_time = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_time = int(datetime.now().timestamp() * 1000)
    current_start = start_time
    
    # ============================================================
    # 1. 直接從 Secrets 讀取 WebShare Proxy (不用迴圈測試)
    # ============================================================
    try:
        proxy_url = st.secrets["general"]["binance_proxy"]
        working_proxy = {
            "http": proxy_url,
            "https": proxy_url
        }
        print(f"🚀 使用 WebShare Proxy 連線...")
    except Exception:
        st.error("❌ 尚未設定 binance_proxy！請檢查 secrets.toml")
        return pd.DataFrame()

    # 2. 設定偽裝 Headers (避免被防火牆擋)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # 3. 開始抓取迴圈
    while current_start < end_time:
        params = {"symbol": symbol, "interval": interval, "startTime": current_start, "limit": 1000}
        try:
            # 加入 proxies 參數
            resp = requests.get(url, params=params, headers=headers, proxies=working_proxy, timeout=10)
            
            # 如果被擋 (403/451)，代表這個 Proxy IP 是美國的，或者被 Ban 了
            if resp.status_code != 200:
                st.error(f"❌ Proxy 連線被拒 (Code {resp.status_code})。請確認 WebShare IP 地區非美國。")
                print(f"❌ API Error: {resp.status_code} - {resp.text}")
                break

            data = resp.json()
            if not data: break
            
            # 防呆：檢查是否回傳錯誤訊息
            if isinstance(data, dict) and 'code' in data: 
                print(f"❌ Binance Error: {data}")
                break
                
            all_data.extend(data)
            current_start = data[-1][6] + 1
            
            # 稍微休息，避免太快把 WebShare 的流量用完或觸發 Binance 限制
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"❌ 傳輸中斷: {e}")
            break
    
    if not all_data: return pd.DataFrame()

    # 4. 資料整理
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "nt", "tb", "tq", "ig"]
    df = pd.DataFrame(all_data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
        
    df = df.drop_duplicates(subset=['open_time']).reset_index(drop=True)
    
    print(f"✅ 成功抓取 {len(df)} 筆數據 (最新: {df['open_time'].max()})")
    return df

# --- Google Trends ---
def load_trends_cache():
    if os.path.exists(TRENDS_CACHE_FILE):
        try:
            df = pd.read_csv(TRENDS_CACHE_FILE)
            df['date_trends'] = pd.to_datetime(df['date_trends'])
            return df
        except: pass
    return pd.DataFrame()

def fetch_google_trends_enhanced(keywords=["Bitcoin"], days_back=1000):
    cached_df = load_trends_cache()
    if not cached_df.empty:
        cache_latest = cached_df['date_trends'].max()
        days_old = (datetime.now() - cache_latest).days
        if days_old < 7: 
            return cached_df
    return cached_df

# --- Fear & Greed ---
def fetch_fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
            df['fear_greed'] = pd.to_numeric(df['value'])
            return df[['date', 'fear_greed']].sort_values('date').reset_index(drop=True)
    except: pass
    return pd.DataFrame()

# --- Wikipedia ---
def fetch_wikipedia_views_history(days_back=365):
    if os.path.exists(WIKI_CACHE_FILE):
        try:
            df = pd.read_csv(WIKI_CACHE_FILE)
            df['date_wiki'] = pd.to_datetime(df['date_wiki'])
            last_date = df['date_wiki'].max()
            if (datetime.now() - last_date).days < 2:
                return df
        except:
            pass

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    headers = {'User-Agent': 'BitcoinRiskDashboard/1.0'}
    article = "Bitcoin"
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/daily/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'items' in data:
                data_list = []
                for item in data['items']:
                    timestamp = item['timestamp']
                    date_obj = datetime.strptime(timestamp[:8], '%Y%m%d')
                    views = item['views']
                    data_list.append({'date_wiki': date_obj, 'wiki_views': views})
                
                df = pd.DataFrame(data_list)
                
                if os.path.exists(WIKI_CACHE_FILE):
                    try:
                        old_df = pd.read_csv(WIKI_CACHE_FILE)
                        old_df['date_wiki'] = pd.to_datetime(old_df['date_wiki'])
                        df = pd.concat([old_df, df]).drop_duplicates(subset=['date_wiki']).sort_values('date_wiki')
                    except: pass

                df.to_csv(WIKI_CACHE_FILE, index=False)
                return df
    except Exception as e:
        pass

    if os.path.exists(WIKI_CACHE_FILE):
        try:
            df = pd.read_csv(WIKI_CACHE_FILE)
            df['date_wiki'] = pd.to_datetime(df['date_wiki'])
            return df
        except: pass
        
    return pd.DataFrame()

# --- Blockchain.com ---
def fetch_blockchain_com_stats(days_back=365):
    try:
        url = "https://api.blockchain.info/charts/n-unique-addresses?timespan=2year&format=json&cors=true"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            data_list = [{'date': pd.to_datetime(p['x'], unit='s'), 'unique_addresses': p['y']} for p in data['values']]
            df = pd.DataFrame(data_list)
            df.rename(columns={'date': 'date_blockchain'}, inplace=True)
            return df
    except: pass
    return pd.DataFrame()

# --- Coinglass ---
def fetch_coinglass_sentiment(days_back=90):
    try:
        url = "https://open-api.coinglass.com/public/v2/funding"
        params = {'symbol': 'BTC', 'interval': '8h'}
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data:
                data_list = [{'date_coinglass': pd.to_datetime(i['time'], unit='ms'), 'funding_rate': float(i.get('rate', 0))} for i in data['data']]
                return pd.DataFrame(data_list)
    except: pass
    return pd.DataFrame()

# --- Obituaries ---
def fetch_bitcoin_obituaries():
    backup_data = [
        {'date_obituaries': pd.to_datetime('2022-11-01'), 'obituary_count': 18},
        {'date_obituaries': pd.to_datetime('2023-01-01'), 'obituary_count': 5},
        {'date_obituaries': pd.to_datetime('2024-01-01'), 'obituary_count': 2},
        {'date_obituaries': pd.to_datetime('2024-06-01'), 'obituary_count': 1},
    ]
    return pd.DataFrame(backup_data)

# --- CMC ---
def fetch_cmc_trending():
    try:
        return 5 
    except: return 50

# --- Google News ---
def fetch_google_news_mentions(days_back=90):
    try:
        import feedparser
        url = "https://news.google.com/rss/search?q=Bitcoin+when:30d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if len(feed.entries) > 0:
            data_list = [{'date': datetime(*entry.published_parsed[:6]).date(), 'title': entry.title} for entry in feed.entries if entry.get('published_parsed')]
            df = pd.DataFrame(data_list)
            df_daily = df.groupby('date').size().reset_index(name='news_count')
            df_daily['date_news'] = pd.to_datetime(df_daily['date'])
            return df_daily
    except: pass
    return pd.DataFrame()

# --- App Store Rank ---
def fetch_coinbase_rank():
    """使用最穩定的 iTunes RSS 接口抓取排名"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. 總榜免費 App (前 100 名)
    overall_url = "https://itunes.apple.com/us/rss/topfreeapplications/limit=100/json"
    # 2. 財務類別免費 App (前 100 名, Genre 6005 是 Finance)
    finance_url = "https://itunes.apple.com/us/rss/topfreeapplications/limit=100/genre=6005/json"
    
    try:
        # 獲取總榜資料
        overall_resp = requests.get(overall_url, headers=headers, timeout=10)
        overall_data = overall_resp.json()
        overall_entries = overall_data.get('feed', {}).get('entry', [])
        
        # 獲取財務類別資料
        finance_resp = requests.get(finance_url, headers=headers, timeout=10)
        finance_data = finance_resp.json()
        finance_entries = finance_data.get('feed', {}).get('entry', [])

        # 提取 App 名稱列表
        top_apps = [entry.get('im:name', {}).get('label', '') for entry in overall_entries]
        finance_apps = [entry.get('im:name', {}).get('label', '') for entry in finance_entries]
        
        # 找尋總榜排名 (1-100)
        overall_rank = None
        for i, name in enumerate(top_apps):
            if "coinbase" in name.lower():
                overall_rank = i + 1
                break
        
        # 找尋財務類別排名 (1-100)
        finance_rank = None
        for i, name in enumerate(finance_apps):
            if "coinbase" in name.lower():
                finance_rank = i + 1
                break
                
        return {
            "overall": overall_rank,
            "finance": finance_rank,
            "top_apps": top_apps
        }
    except Exception as e:
        st.error(f"App Store 資料更新失敗: {e}")
        return None
    
# --- Rainbow Chart ---
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

# --- YouTube Data Management ---
def load_youtube_activity_history():
    if os.path.exists(YOUTUBE_CACHE_FILE):
        try:
            df = pd.read_csv(YOUTUBE_CACHE_FILE)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except: pass
    return pd.DataFrame()

from scipy.stats import pearsonr

def find_similar_patterns(df, lookback_days=60, forecast_days=30):
    """
    尋找與當前市場最相似的歷史片段
    """
    if len(df) < lookback_days + forecast_days:
        return []

    # 準備當前數據 (Target)
    current_slice = df.iloc[-lookback_days:].copy()
    
    # 1. 提取特徵：風險曲線
    target_risk = current_slice['total_risk'].values
    
    # 2. 提取特徵：價格走勢 (使用 Min-Max Normalization 讓形狀可比對)
    target_price = current_slice['close'].values
    target_price_norm = (target_price - target_price.min()) / (target_price.max() - target_price.min())

    results = []

    # 3. 滑動窗口遍歷歷史 (避開最近這段時間，以免自己比對自己)
    # search_end_index 是為了預留 forecast_days 的空間看"未來"
    search_end_index = len(df) - lookback_days - forecast_days 
    
    for i in range(0, search_end_index, 2): # step=2 加速計算
        # 歷史片段
        hist_slice = df.iloc[i : i + lookback_days]
        
        # 歷史特徵
        hist_risk = hist_slice['total_risk'].values
        hist_price = hist_slice['close'].values
        
        # 簡單過濾：如果風險值差異過大(例如現在是高風險，卻比對到低風險區)，直接跳過
        if abs(hist_risk.mean() - target_risk.mean()) > 0.3:
            continue

        hist_price_norm = (hist_price - hist_price.min()) / (hist_price.max() - hist_price.min())

        # 計算相關性 (Correlation)
        # 我們綜合考量「風險曲線相似度」和「價格型態相似度」
        try:
            corr_risk, _ = pearsonr(target_risk, hist_risk)
            corr_price, _ = pearsonr(target_price_norm, hist_price_norm)
            
            # 綜合分數 (你可以調整權重，這裡假設風險指標形狀更重要)
            final_score = (corr_risk * 0.7) + (corr_price * 0.3)
            
            if final_score > 0.80: # 只保留高度相似的
                # 紀錄這段歷史發生後的"未來"漲跌幅
                future_slice = df.iloc[i + lookback_days : i + lookback_days + forecast_days]
                future_return = (future_slice['close'].iloc[-1] - hist_price[-1]) / hist_price[-1]
                
                results.append({
                    'start_date': hist_slice['open_time'].iloc[0],
                    'end_date': hist_slice['open_time'].iloc[-1],
                    'score': final_score,
                    'future_return': future_return,
                    'hist_price_data': hist_price, # 原始價格
                    'hist_risk_data': hist_risk,
                    'future_price_data': future_slice['close'].values
                })
        except:
            continue

    # 根據分數排序，取前 5 名
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    return results, target_price, target_risk
# ============================================================
# BBW 波動率預警功能 (含歷史標記)
# ============================================================
def get_bbw_squeeze_chart(df, lookback=20, std_dev=2.0, squeeze_threshold=0.05):
    # 1. 計算布林通道與 BBW
    df_bb = df.copy()
    df_bb['SMA'] = df_bb['close'].rolling(window=lookback).mean()
    df_bb['std'] = df_bb['close'].rolling(window=lookback).std()
    
    df_bb['Upper'] = df_bb['SMA'] + (df_bb['std'] * std_dev)
    df_bb['Lower'] = df_bb['SMA'] - (df_bb['std'] * std_dev)
    
    # BBW 公式: (上軌 - 下軌) / 中軌
    df_bb['BBW'] = (df_bb['Upper'] - df_bb['Lower']) / df_bb['SMA']
    
    # 為了圖表清晰，取 2019 之後
    df_bb = df_bb[df_bb['open_time'] >= "2019-01-01"]
    
    # 2. 準備繪圖
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3], 
        subplot_titles=("布林通道與歷史變盤點 (Price & Squeeze Signals)", "BBW 波動率寬度")
    )

    # --- 上圖: 價格 + 通道 ---
    fig.add_trace(go.Scatter(
        x=df_bb['open_time'], y=df_bb['close'],
        mode='lines', name='Price',
        line=dict(color='white', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_bb['open_time'], y=df_bb['Upper'],
        mode='lines', name='Upper',
        line=dict(width=0), showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_bb['open_time'], y=df_bb['Lower'],
        mode='lines', name='Lower',
        line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0, 229, 255, 0.1)',
        showlegend=False
    ), row=1, col=1)

    # --- 【新增功能】標記歷史變盤點 ---
    # 找出所有符合壓縮條件的時間點
    squeeze_points = df_bb[df_bb['BBW'] < squeeze_threshold].copy()
    
    if not squeeze_points.empty:
        # 根據當時的風險值決定顏色
        # 邏輯：低風險壓縮=潛在買點(綠)，高風險壓縮=潛在賣點(紅)
        colors = []
        hover_texts = []
        
        for idx, row in squeeze_points.iterrows():
            risk = row['total_risk']
            bbw_val = row['BBW']
            
            if risk < 0.4:
                colors.append('#00e676') # 綠色 (看漲蓄力)
                bias = "Bullish Setup"
            elif risk > 0.6:
                colors.append('#ff1744') # 紅色 (看跌蓄力)
                bias = "Bearish Setup"
            else:
                colors.append('#ffea00') # 黃色 (中性)
                bias = "Neutral Squeeze"
                
            hover_texts.append(f"<b>{bias}</b><br>Date: {row['open_time'].strftime('%Y-%m-%d')}<br>BBW: {bbw_val:.4f}<br>Risk: {risk:.2f}")

        fig.add_trace(go.Scatter(
            x=squeeze_points['open_time'],
            y=squeeze_points['close'],
            mode='markers',
            name='歷史變盤訊號',
            marker=dict(
                color=colors,
                size=8,
                symbol='diamond-open', # 空心菱形，比較不擋視線
                line=dict(width=2, color=colors)
            ),
            hoverinfo='text',
            hovertext=hover_texts
        ), row=1, col=1)

    # --- 下圖: BBW 指標 ---
    bbw_colors = np.where(df_bb['BBW'] < squeeze_threshold, '#ffea00', '#2979ff')
    
    fig.add_trace(go.Bar(
        x=df_bb['open_time'], y=df_bb['BBW'],
        name='BBW', marker_color=bbw_colors, opacity=0.8
    ), row=2, col=1)

    fig.add_hline(
        y=squeeze_threshold, 
        line_dash="dash", line_color="red", line_width=1,
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark", height=650, hovermode='x unified', showlegend=True,
        yaxis=dict(title="Price", type='log'), yaxis2=dict(title="BBW")
    )

    return fig, df_bb.iloc[-1]['BBW']
# ============================================================
# 歷史波動率 (Historical Volatility) - 暴風雨前的寧靜
# ============================================================
def get_historical_volatility_chart(df, window=30, threshold=25):
    df_hv = df.copy()
    
    # 1. 計算對數收益率 (Log Returns)
    df_hv['log_ret'] = np.log(df_hv['close'] / df_hv['close'].shift(1))
    
    # 2. 計算滾動標準差 (Rolling Std Dev)
    # 3. 年化處理 (Annualize): 乘以 sqrt(365)
    df_hv['hv'] = df_hv['log_ret'].rolling(window=window).std() * np.sqrt(365) * 100
    
    # 取 2017 之後
    plot_data = df_hv[df_hv['open_time'] >= "2017-08-17"].copy()
    
    # 4. 繪圖
    fig = go.Figure()
    
    # HV 曲線
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['hv'],
        mode='lines', name=f'{window}D Historical Volatility',
        line=dict(color='#00e5ff', width=1.5)
    ))
    
    # 警戒線 (低波動閾值)
    fig.add_hline(
        y=threshold, 
        line_dash="dash", line_color="#ff1744", line_width=1,
        annotation_text=f"極度壓縮區 (<{threshold}%)", 
        annotation_position="bottom right"
    )
    
    # 標記低波動區域 (變盤前夕)
    low_vol_mask = plot_data['hv'] < threshold
    # 為了視覺效果，我們只標記連續低波動的區段，或者用背景色填充
    # 這裡簡單用紅色圓點標記低於閾值的時刻
    low_vol_points = plot_data[low_vol_mask]
    
    fig.add_trace(go.Scatter(
        x=low_vol_points['open_time'], y=low_vol_points['hv'],
        mode='markers', name='壓縮訊號 (Squeeze)',
        marker=dict(color='#ff1744', size=4, symbol='circle'),
        opacity=0.6
    ))

    # 當前狀態
    curr_hv = plot_data.iloc[-1]['hv']
    
    if curr_hv < threshold:
        status = "⚡ 暴風雨前的寧靜 (極度壓縮)"
        desc = "波動率觸底，大行情即將爆發，請留意突破方向！"
        color = "#ff1744"
    elif curr_hv > 80:
        status = "🌊 巨浪滔天 (高波動)"
        desc = "市場情緒激動，風險與機會並存。"
        color = "#00e676"
    else:
        status = "💨 波動正常"
        desc = "市場處於常態波動區間。"
        color = "#ffffff"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.95,
        text=f"<b>Current HV: {curr_hv:.2f}%</b><br><span style='color:{color}'>{status}</span><br><span style='font-size:10px;color:gray'>{desc}</span>",
        showarrow=False,
        bgcolor="rgba(0,0,0,0.8)", bordercolor=color, borderwidth=1,
        font=dict(size=14, color="white"),
        align="left"
    )

    fig.update_layout(
        title=f"Bitcoin Historical Volatility (HV{window})",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        yaxis=dict(title="Annualized Volatility (%)"),
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig, curr_hv
# ============================================================
# TTM Squeeze (鑽石訊號版) - 仿 BBW 風格
# ============================================================
def get_ttm_squeeze_chart(df, length=20, mult=2.0, length_kc=20, mult_kc=1.5):
    df_ttm = df.copy()
    
    # 1. 計算布林通道
    df_ttm['basis'] = df_ttm['close'].rolling(window=length).mean()
    df_ttm['dev'] = df_ttm['close'].rolling(window=length).std()
    df_ttm['upper_bb'] = df_ttm['basis'] + (df_ttm['dev'] * mult)
    df_ttm['lower_bb'] = df_ttm['basis'] - (df_ttm['dev'] * mult)

    # 2. 計算肯特納通道
    df_ttm['tr0'] = abs(df_ttm['high'] - df_ttm['low'])
    df_ttm['tr1'] = abs(df_ttm['high'] - df_ttm['close'].shift(1))
    df_ttm['tr2'] = abs(df_ttm['low'] - df_ttm['close'].shift(1))
    df_ttm['tr'] = df_ttm[['tr0', 'tr1', 'tr2']].max(axis=1)
    df_ttm['atr'] = df_ttm['tr'].rolling(window=length_kc).mean()
    
    df_ttm['upper_kc'] = df_ttm['basis'] + (df_ttm['atr'] * mult_kc)
    df_ttm['lower_kc'] = df_ttm['basis'] - (df_ttm['atr'] * mult_kc)

    # 3. 判斷擠壓狀態 (Squeeze On)
    df_ttm['squeeze_on'] = (df_ttm['upper_bb'] < df_ttm['upper_kc']) & (df_ttm['lower_bb'] > df_ttm['lower_kc'])
    
    # 判斷 "點火" (Squeeze Off 的第一天)
    # 也就是：昨天是 True，今天是 False
    df_ttm['fired'] = (df_ttm['squeeze_on'].shift(1) == True) & (df_ttm['squeeze_on'] == False)

    # 4. 計算動能 (Momentum)
    df_ttm['avg_price'] = (df_ttm['high'] + df_ttm['low']) / 2
    df_ttm['delta'] = df_ttm['close'] - (df_ttm['avg_price'] + df_ttm['basis']) / 2
    df_ttm['momentum'] = df_ttm['delta'].rolling(window=length).mean() * 5 

    # 取 2021 之後數據
    plot_data = df_ttm[df_ttm['open_time'] >= "2021-01-01"].copy()

    # 5. 繪圖
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3], 
        subplot_titles=("價格與擠壓訊號 (Price & Squeeze Signals)", "動能方向 (Momentum)")
    )

    # --- 上圖: 價格線 ---
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['close'],
        mode='lines', name='Price',
        line=dict(color='#333333', width=1.5) # 深灰色線條
    ), row=1, col=1)

    # 畫出肯特納通道邊界 (淡淡的參考)
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['upper_kc'],
        mode='lines', line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'), showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['lower_kc'],
        mode='lines', line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'), showlegend=False,
        fill='tonexty', fillcolor='rgba(0,0,0,0.03)'
    ), row=1, col=1)

    # --- 【關鍵修改】圖標訊號 (Diamonds on Price) ---
    
    # 1. 擠壓中 (紅色鑽石)
    squeeze_points = plot_data[plot_data['squeeze_on']]
    if not squeeze_points.empty:
        fig.add_trace(go.Scatter(
            x=squeeze_points['open_time'], 
            y=squeeze_points['close'],
            mode='markers', 
            name='蓄力中 (Squeeze)',
            marker=dict(
                symbol='diamond', # 實心菱形
                size=7, 
                color='#d32f2f', # 深紅色
                line=dict(width=1, color='white') # 白邊增加對比
            ),
            hoverinfo='x+y+name'
        ), row=1, col=1)

    # 2. 點火發射 (綠色鑽石 - 只有變盤那天顯示)
    fired_points = plot_data[plot_data['fired']]
    if not fired_points.empty:
        fig.add_trace(go.Scatter(
            x=fired_points['open_time'], 
            y=fired_points['close'],
            mode='markers', 
            name='🔥 點火 (Fired)',
            marker=dict(
                symbol='star', # 星形代表爆發
                size=12, 
                color='#00c853', # 鮮綠色
                line=dict(width=1, color='black')
            ),
            hoverinfo='x+y+name'
        ), row=1, col=1)

    # --- 下圖: 動能柱 ---
    colors = []
    prev_m = 0
    for m in plot_data['momentum']:
        if m >= 0:
            colors.append('#00897b' if m > prev_m else '#80cbc4') # 深青 vs 淺青
        else:
            colors.append('#e53935' if m < prev_m else '#ef9a9a') # 深紅 vs 淺紅
        prev_m = m

    fig.add_trace(go.Bar(
        x=plot_data['open_time'], y=plot_data['momentum'],
        name='Momentum',
        marker_color=colors
    ), row=2, col=1)

    # 7. 狀態解讀
    last_sqz = plot_data.iloc[-1]['squeeze_on']
    last_mom = plot_data.iloc[-1]['momentum']
    prev_mom = plot_data.iloc[-2]['momentum']
    
    if last_sqz:
        status = "🔴 壓縮蓄力中 (SQUEEZE ON)"
        desc = "圖表出現紅色菱形 ♦，波動極低，等待大行情。"
        s_color = "#d32f2f"
    else:
        status = "🟢 能量釋放中 (ACTIVE)"
        desc = "趨勢運行中。"
        s_color = "#388e3c"

    if last_mom > 0:
        mom_text = "📈 動能向上 (多方)"
    else:
        mom_text = "📉 動能向下 (空方)"

    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=f"<b>{status}</b><br><span style='font-size:12px;color:#555'>{desc}</span><br><span style='font-size:14px;color:#333'>{mom_text}</span>",
        showarrow=False, bgcolor="rgba(255,255,255,0.9)", bordercolor=s_color, borderwidth=2,
        font=dict(size=16, color="#333"), align="left"
    )

    fig.update_layout(
        template="plotly_white", 
        height=600, 
        hovermode='x unified',
        yaxis=dict(title="Price", type='log', autorange=True),
        yaxis2=dict(title="Momentum"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig, last_sqz

# ============================================================
# Ehlers Fisher Transform (費雪變換) - 極值轉折偵測
# ============================================================
def get_ehlers_fisher_chart(df, length=10):
    df_fish = df.copy().reset_index(drop=True)
    
    # 1. 準備數據: 計算中價 (High+Low)/2
    highs = df_fish['high'].values
    lows = df_fish['low'].values
    mids = (highs + lows) / 2
    
    # 初始化陣列
    n = len(df_fish)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    value1 = np.zeros(n)
    
    # 2. 遞迴計算 (Ehlers 原始算法)
    # Value1 = 0.33 * 2 * ((Mid - MinLow) / (MaxHigh - MinLow) - 0.5) + 0.67 * PrevValue1
    # Fisher = 0.5 * 0.5 * ln((1 + Value1) / (1 - Value1)) + 0.5 * PrevFisher
    
    for i in range(length, n):
        # 找出過去 Length 天的最高與最低中價
        min_l = np.min(lows[i-length+1 : i+1])
        max_h = np.max(highs[i-length+1 : i+1])
        
        # 防止分母為 0
        div = max_h - min_l
        if div == 0: div = 0.001
            
        # 正規化價格 (-1 到 1 之間)
        v1 = 0.33 * 2 * ((mids[i] - min_l) / div - 0.5) + 0.67 * value1[i-1]
        
        # 限制邊界，防止 Log 爆掉 (必須小於 1)
        if v1 > 0.99: v1 = 0.999
        if v1 < -0.99: v1 = -0.999
        value1[i] = v1
        
        # 計算 Fisher
        fisher[i] = 0.5 * 0.5 * np.log((1 + v1) / (1 - v1)) + 0.5 * fisher[i-1]
        
        # Trigger 線 (就是 Fisher 延遲一根)
        trigger[i] = fisher[i-1]

    df_fish['fisher'] = fisher
    df_fish['trigger'] = trigger

    # 取 2021 之後
    plot_data = df_fish[df_fish['open_time'] >= "2021-01-01"].copy()

    # 3. 繪圖
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.6, 0.4], 
        subplot_titles=("Price Action", "Ehlers Fisher Transform")
    )

    # --- 上圖: 價格 ---
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['close'],
        mode='lines', name='Price',
        line=dict(color='#333333', width=1.5)
    ), row=1, col=1)

    # 標記交叉轉折點 (Crossovers)
    # 金叉：Fisher 上穿 Trigger (買)
    # 死叉：Fisher 下穿 Trigger (賣)
    # 為了過濾雜訊，我們通常只看 "極值區" 的交叉 (例如 >1.5 或 <-1.5)
    
    cross_buy = (plot_data['fisher'] > plot_data['trigger']) & \
                (plot_data['fisher'].shift(1) <= plot_data['trigger'].shift(1)) & \
                (plot_data['fisher'] < -1.0) # 只有在底部交叉才算抄底

    cross_sell = (plot_data['fisher'] < plot_data['trigger']) & \
                 (plot_data['fisher'].shift(1) >= plot_data['trigger'].shift(1)) & \
                 (plot_data['fisher'] > 1.0) # 只有在頂部交叉才算逃頂

    if cross_buy.any():
        pts = plot_data[cross_buy]
        fig.add_trace(go.Scatter(
            x=pts['open_time'], y=pts['low']*0.96,
            mode='markers', name='Fisher Buy',
            marker=dict(symbol='triangle-up', size=9, color='#2979ff')
        ), row=1, col=1)

    if cross_sell.any():
        pts = plot_data[cross_sell]
        fig.add_trace(go.Scatter(
            x=pts['open_time'], y=pts['high']*1.04,
            mode='markers', name='Fisher Sell',
            marker=dict(symbol='triangle-down', size=9, color='#ff1744')
        ), row=1, col=1)

    # --- 下圖: Fisher 指標 ---
    # 根據趨勢變色
    # Fisher > Trigger = 上漲趨勢 (綠)
    # Fisher < Trigger = 下跌趨勢 (紅)
    
    # 為了畫出顏色區塊，我們用 fill='tonexty' 技巧
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['fisher'],
        mode='lines', name='Fisher',
        line=dict(color='#00e676', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['trigger'],
        mode='lines', name='Trigger',
        line=dict(color='#ff1744', width=1.5, dash='dot'),
        fill='tonexty', # 填滿兩線之間
        fillcolor='rgba(0, 230, 118, 0.1)' # 預設底色
    ), row=2, col=1)

    # 繪製極值線 (2.0 和 -2.0)
    fig.add_hline(y=2.0, line_dash="dot", line_color="gray", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=-2.0, line_dash="dot", line_color="gray", row=2, col=1, annotation_text="Oversold")

    # 4. 狀態解讀
    curr_fish = plot_data.iloc[-1]['fisher']
    curr_trig = plot_data.iloc[-1]['trigger']
    prev_fish = plot_data.iloc[-2]['fisher']
    
    # 判斷轉折
    is_turning_up = (curr_fish > curr_trig) and (prev_fish <= plot_data.iloc[-2]['trigger'])
    is_turning_down = (curr_fish < curr_trig) and (prev_fish >= plot_data.iloc[-2]['trigger'])
    
    if curr_fish > 2.0:
        status = "🔥 極度過熱 (準備反轉向下)"
        clr = "#d50000"
    elif curr_fish < -2.0:
        status = "🧊 極度超跌 (準備反轉向上)"
        clr = "#2979ff"
    elif curr_fish > curr_trig:
        status = "↗️ 多頭趨勢中"
        clr = "#00c853"
    else:
        status = "↘️ 空頭趨勢中"
        clr = "#ff5722"

    # 如果剛好轉折，覆蓋狀態
    if is_turning_up: status = "⚡ 黃金交叉 (買點確認)!"
    if is_turning_down: status = "⚡ 死亡交叉 (賣點確認)!"

    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=f"<b>Fisher: {curr_fish:.2f}</b><br><span style='color:{clr}'>{status}</span>",
        showarrow=False, bgcolor="rgba(255,255,255,0.9)", bordercolor=clr, borderwidth=2,
        font=dict(size=14, color="#333"), align="left"
    )

    fig.update_layout(
        template="plotly_white", height=600, hovermode='x unified',
        yaxis=dict(title="Price", type='log', autorange=True),
        yaxis2=dict(title="Fisher Score", range=[plot_data['fisher'].min()-0.5, plot_data['fisher'].max()+0.5]),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig, curr_fish
    
# ============================================================
# Coppock Curve (終極版 - 包含大底與大頂)
# ============================================================
def get_coppock_curve_chart(df, wma_len=10, roc1_len=14, roc2_len=11, bottom_threshold=-10, top_threshold=15):
    df_cc = df.copy()
    
    # 1. 計算 ROC
    roc1 = df_cc['close'].diff(roc1_len) / df_cc['close'].shift(roc1_len) * 100
    roc2 = df_cc['close'].diff(roc2_len) / df_cc['close'].shift(roc2_len) * 100
    
    roc_sum = roc1 + roc2
    
    # 2. 計算 WMA
    def calc_wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    df_cc['coppock'] = calc_wma(roc_sum, wma_len)
    
    # 取 2018 之後
    plot_data = df_cc[df_cc['open_time'] >= "2018-01-01"].copy()

    # 3. 繪圖
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.6, 0.4], 
        subplot_titles=("Price Action", "Coppock Curve (Cycle Top & Bottom)")
    )

    # --- 上圖: K 線圖 ---
    fig.add_trace(go.Candlestick(
        x=plot_data['open_time'],
        open=plot_data['open'], high=plot_data['high'],
        low=plot_data['low'], close=plot_data['close'],
        name='Price'
    ), row=1, col=1)

    # --- 下圖: Coppock Curve ---
    cc_vals = plot_data['coppock'].values
    colors = []
    
    for i in range(len(cc_vals)):
        curr = cc_vals[i]
        prev = cc_vals[i-1] if i > 0 else 0
        
        if curr >= 0:
            if curr > prev: colors.append("#00e676") # 強多 (深綠)
            else: colors.append("#b9f6ca") # 多頭減弱 (淺綠)
        else:
            if curr > prev: colors.append("#ff80ab") # 底部反彈 (粉紅)
            else: colors.append("#d50000") # 空頭 (深紅)
            
    fig.add_trace(go.Bar(
        x=plot_data['open_time'], y=plot_data['coppock'],
        name='Coppock',
        marker_color=colors
    ), row=2, col=1)

    # --- 訊號偵測 (買點 & 賣點) ---
    buy_signals, buy_dates, buy_prices = [], [], []
    sell_signals, sell_dates, sell_prices = [], [], []
    
    # 冷卻機制 (避免連續箭頭)
    last_buy_idx = -999
    last_sell_idx = -999
    
    for i in range(5, len(plot_data)):
        curr = plot_data['coppock'].iloc[i]
        prev = plot_data['coppock'].iloc[i-1]
        prev2 = plot_data['coppock'].iloc[i-2]
        
        # === 🟢 大底偵測 (Deep Buy) ===
        # 條件：數值夠低 + V 型反轉
        if curr < bottom_threshold and prev < bottom_threshold and curr > prev and prev < prev2:
            if i - last_buy_idx > 20: # 買入冷卻
                buy_signals.append(curr)
                buy_dates.append(plot_data['open_time'].iloc[i])
                buy_prices.append(plot_data['low'].iloc[i])
                last_buy_idx = i

        # === 🟣 大頂偵測 (Great Top) ===
        # 條件：數值夠高 + 倒 V 型反轉 (M頭)
        if curr > top_threshold and prev > top_threshold and curr < prev and prev > prev2:
            if i - last_sell_idx > 20: # 賣出冷卻
                sell_signals.append(curr)
                sell_dates.append(plot_data['open_time'].iloc[i])
                sell_prices.append(plot_data['high'].iloc[i])
                last_sell_idx = i

    # 繪製買點 (粉紅向上箭頭)
    if buy_signals:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=np.array(buy_prices) * 0.9,
            mode='markers', name='Deep Buy (大底)',
            marker=dict(symbol='triangle-up', size=14, color='#ff4081', line=dict(width=2, color='white'))
        ), row=1, col=1)

    # 繪製賣點 (青色向下箭頭) - 這裡用青色與紅色區隔
    if sell_signals:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=np.array(sell_prices) * 1.1, # 標在 K 線上方
            mode='markers', name='Great Top (大頂)',
            marker=dict(symbol='triangle-down', size=14, color='#00e5ff', line=dict(width=2, color='white'))
        ), row=1, col=1)

    # 4. 繪製閾值參考線
    fig.add_hline(y=bottom_threshold, line_dash="dot", line_color="#ff4081", row=2, col=1, annotation_text="Buy Zone")
    fig.add_hline(y=top_threshold, line_dash="dot", line_color="#00e5ff", row=2, col=1, annotation_text="Sell Zone")
    fig.add_hline(y=0, line_color="white", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", 
        height=600, 
        hovermode='x unified',
        yaxis=dict(title="Price", type='log', autorange=True),
        yaxis2=dict(title="Coppock Val"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.02),
        xaxis_rangeslider_visible=False
    )
    
    return fig, plot_data.iloc[-1]['coppock']

# ============================================================
# Pi Cycle Top & 200WMA (週期天花板與地板) - 修復歷史顯示
# ============================================================
def get_cycle_master_chart(df, use_log=True):
    df_cy = df.copy()
    
    # 1. 計算指標 (使用全歷史數據計算，確保訊號準確)
    # Pi Cycle Top components
    df_cy['111SMA'] = df_cy['close'].rolling(window=111).mean()
    df_cy['350SMA_x2'] = df_cy['close'].rolling(window=350).mean() * 2
    
    # 200 Week Moving Average (200 * 7 = 1400 days)
    df_cy['200WMA'] = df_cy['close'].rolling(window=1400).mean()
    
    # 2. 檢測交叉訊號 (Pi Cycle Top Signal)
    # 找出 111SMA 上穿 350SMA_x2 的點 (死亡交叉)
    df_cy['pi_cross'] = (df_cy['111SMA'] > df_cy['350SMA_x2']) & \
                        (df_cy['111SMA'].shift(1) <= df_cy['350SMA_x2'].shift(1))
    
    # 3. 決定顯示範圍
    # 為了能看到 2013/2017 的訊號，我們預設顯示全歷史，或者至少從 2012 開始
    # 這裡我們設定如果數據夠長，就從 2012 開始顯示；否則顯示全部
    start_date_filter = "2012-01-01"
    plot_data = df_cy[df_cy['open_time'] >= start_date_filter].copy()
    
    if plot_data.empty: # 如果數據不足，就用全部
        plot_data = df_cy.copy()

    # 找出顯示範圍內的交叉點
    cross_points = plot_data[plot_data['pi_cross']]

    # 4. 繪圖
    fig = go.Figure()

    # (A) 價格線
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['close'],
        mode='lines', name='BTC Price',
        line=dict(color='#F7931A', width=1.5)
    ))

    # (B) 200WMA (地板)
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['200WMA'],
        mode='lines', name='200 Week MA (鐵底)',
        line=dict(color='#00e676', width=2)
    ))
    
    # 地板下方的顏色填充
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['200WMA'] * 0.9,
        mode='lines', line=dict(width=0), showlegend=False,
        fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)',
        name='Deep Value Zone'
    ))

    # (C) Pi Cycle Lines
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['111SMA'],
        mode='lines', name='111 SMA (快線)',
        line=dict(color='#ff9100', width=1, dash='dot') # 橘色虛線
    ))
    
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['350SMA_x2'],
        mode='lines', name='350 SMA x2 (慢線)',
        line=dict(color='#ff1744', width=1) # 紅色實線
    ))

    # (D) 標記死亡交叉點 (Pi Cycle Top Signal)
    if not cross_points.empty:
        # 為了避免標籤重疊，我們只標記 "Top"
        fig.add_trace(go.Scatter(
            x=cross_points['open_time'], 
            y=cross_points['close'] * 1.1, # 標記在價格上方一點點
            mode='markers+text',
            name='Pi Top Signal',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            text=["TOP"] * len(cross_points),
            textposition="top center",
            textfont=dict(color='red', size=14, family="Arial Black")
        ))
        
        # 額外畫一條垂直線標示時間點
        for date in cross_points['open_time']:
            fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="rgba(255, 0, 0, 0.5)")

    # 5. 當前狀態計算
    last_price = plot_data.iloc[-1]['close']
    last_200w = plot_data.iloc[-1]['200WMA']
    
    # 處理 NaN (如果數據不足 200週)
    if pd.isna(last_200w):
        dist_to_floor_text = "數據不足計算"
        bg_c = "#000000"
    else:
        dist_to_floor = (last_price - last_200w) / last_200w
        dist_to_floor_text = f"+{dist_to_floor:.1%}"
    
    last_111 = plot_data.iloc[-1]['111SMA']
    last_350 = plot_data.iloc[-1]['350SMA_x2']
    
    if pd.isna(last_111) or pd.isna(last_350):
         status_msg = "數據加載中..."
         bg_c = "#000000"
    elif last_111 > last_350: # 已經交叉
        status_msg = "💀 Pi Cycle 警告：市場處於過熱/頂部區域！"
        bg_c = "#3e2723"
    else:
        # 計算快線距離慢線還差多少 %
        dist_to_top = (last_350 - last_111) / last_350
        status_msg = f"⚖️ 週期中段。距離地板 {dist_to_floor_text} | 距離頂部信號尚遠 ({dist_to_top:.1%})"
        bg_c = "#000000"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.05,
        text=status_msg,
        showarrow=False,
        bgcolor=bg_c, bordercolor="white", borderwidth=1,
        font=dict(size=14, color="white")
    )

    fig.update_layout(
        title="Bitcoin Cycle Master (Pi Cycle Top & 200WMA)",
        template="plotly_dark",
        height=600,
        hovermode='x unified',
        yaxis=dict(title="Price (USDT)", type='log' if use_log else 'linear'),
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig

# ============================================================
# Power Law Corridor (冪律法則通道) - 顯示全歷史修正版
# ============================================================
def get_power_law_chart(df, use_log=True):
    df_pl = df.copy()
    
    # 1. 準備數據
    genesis_date = pd.to_datetime("2009-01-03")
    df_pl['days_since_genesis'] = (df_pl['open_time'] - genesis_date).dt.days
    
    # 過濾掉早期數據避免 log(0)
    df_pl = df_pl[df_pl['days_since_genesis'] > 0]
    
    # 2. 計算冪律模型
    # Price = 10 ^ -17 * days ^ 5.8
    df_pl['power_law_fair'] = 10**-17 * (df_pl['days_since_genesis'] ** 5.8)
    
    # 計算通道
    df_pl['pl_support'] = df_pl['power_law_fair'] * 0.35  # 歷史鐵底
    df_pl['pl_resistance'] = df_pl['power_law_fair'] * 2.5 # 歷史泡沫頂
    df_pl['pl_bubble'] = df_pl['power_law_fair'] * 4.0     # 極端泡沫

    # 【關鍵修正】顯示全歷史，從 2011 年開始 (太早的價格太低，意義不大)
    plot_data = df_pl[df_pl['open_time'] >= "2011-01-01"]

    # 3. 繪圖
    fig = go.Figure()

    # (A) 支撐與壓力帶
    # 上軌 (泡沫區)
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['pl_resistance'],
        mode='lines', name='Bubble Zone (泡沫區)',
        line=dict(color='#ff1744', width=1)
    ))
    
    # 公允價值線 (中軸)
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['power_law_fair'],
        mode='lines', name='Fair Value (公允價值)',
        line=dict(color='#2979ff', width=2, dash='dash') 
    ))
    
    # 下軌 (鐵底區)
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['pl_support'],
        mode='lines', name='Bottom Zone (物理鐵底)',
        line=dict(color='#00e676', width=1),
        fill='tonexty', fillcolor='rgba(0, 230, 118, 0.05)' 
    ))

    # (B) 價格線
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['close'],
        mode='lines', name='BTC Price',
        line=dict(color='#F7931A', width=2) 
    ))

    # 4. 狀態判斷
    last_price = plot_data.iloc[-1]['close']
    last_fair = plot_data.iloc[-1]['power_law_fair']
    deviation = (last_price - last_fair) / last_fair
    
    if deviation < -0.3:
        status_msg = "💎 價格低於公允價值，處於積累區！"
        color_s = "#00e676"
    elif deviation > 1.0:
        status_msg = "🔥 價格大幅高於公允價值，注意風險！"
        color_s = "#ff1744"
    else:
        status_msg = "⚖️ 價格回歸公允價值附近 (Fair Value)。"
        color_s = "#ffffff"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        text=f"<b>Power Law Status:</b><br>{status_msg}<br>(Dev: {deviation:.1%})",
        showarrow=False,
        bgcolor="rgba(0,0,0,0.7)", bordercolor=color_s, borderwidth=1,
        font=dict(size=13, color="white"),
        align="left"
    )

    fig.update_layout(
        title="Bitcoin Power Law Corridor (比特幣冪律法則通道)",
        template="plotly_dark",
        height=600,
        hovermode='x unified',
        yaxis=dict(title="Price (USDT)", type='log' if use_log else 'linear'),
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig

# AHR999 囤幣指標 (改良版 - 基於 Power Law) - 顯示全歷史修正版
# ============================================================
def get_ahr999_chart(df):
    df_ahr = df.copy()
    
    # 1. 準備 Power Law Fair Value
    genesis_date = pd.to_datetime("2009-01-03")
    df_ahr['days_since_genesis'] = (df_ahr['open_time'] - genesis_date).dt.days
    df_ahr = df_ahr[df_ahr['days_since_genesis'] > 0]
    
    # Power Law Fair Value
    df_ahr['fair_value'] = 10**-17 * (df_ahr['days_since_genesis'] ** 5.8)
    
    # 2. 計算 200日 幾何平均
    df_ahr['log_price'] = np.log(df_ahr['close'])
    df_ahr['200_geo_mean'] = np.exp(df_ahr['log_price'].rolling(window=200).mean())
    
    # 3. 計算 AHR999
    df_ahr['ahr999'] = (df_ahr['close'] / df_ahr['200_geo_mean']) * (df_ahr['close'] / df_ahr['fair_value'])
    
    # 【關鍵修正】顯示全歷史，從 2011 年開始
    plot_data = df_ahr[df_ahr['open_time'] >= "2011-01-01"].copy()

    # 4. 繪圖
    fig = go.Figure()

    # (A) 繪製 AHR999 線
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['ahr999'],
        mode='lines', name='AHR999 Index',
        line=dict(color='#ffffff', width=2)
    ))

    # (B) 繪製區間 (使用形狀填充)
    # 抄底區 (< 0.45)
    fig.add_hrect(
        y0=0, y1=0.45, 
        fillcolor="#00e676", opacity=0.15, 
        layer="below", line_width=0,
        annotation_text="抄底區 (Buy The Dip)", annotation_position="top left"
    )
    
    # 定投區 (0.45 - 1.2)
    fig.add_hrect(
        y0=0.45, y1=1.2, 
        fillcolor="#2979ff", opacity=0.1, 
        layer="below", line_width=0,
        annotation_text="定投區 (DCA Zone)", annotation_position="top left"
    )
    
    # 起飛/賣出區 (> 1.2)
    fig.add_hrect(
        y0=1.2, y1=100, 
        fillcolor="#ff1744", opacity=0.1, 
        layer="below", line_width=0,
        annotation_text="起飛/泡沫區 (Hold/Sell)", annotation_position="bottom left"
    )

    # 關鍵線
    fig.add_hline(y=0.45, line_dash="dash", line_color="#00e676", line_width=1)
    fig.add_hline(y=1.2, line_dash="dash", line_color="#ff1744", line_width=1)

    # 當前狀態標註
    curr_ahr = plot_data.iloc[-1]['ahr999']
    if curr_ahr < 0.45:
        status = "💎 抄底區 (Bottom)"
        color = "#00e676"
    elif curr_ahr < 1.2:
        status = "👌 定投區 (DCA)"
        color = "#2979ff"
    else:
        status = "🚀 起飛區 (Top)"
        color = "#ff1744"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        text=f"<b>Current AHR999: {curr_ahr:.3f}</b><br>{status}",
        showarrow=False,
        bgcolor="rgba(0,0,0,0.8)", bordercolor=color, borderwidth=2,
        font=dict(size=14, color=color)
    )

    fig.update_layout(
        title="AHR999 Hoarding Index (囤幣指標)",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        yaxis=dict(title="Index Value", type='log', range=[-1, 2]), 
        showlegend=False
    )
    
    return fig, curr_ahr

# ============================================================
# 宏觀指標：2月線 Stoch RSI (雙月隨機相對強弱指數) - 完美標註版
# ============================================================
def get_2m_stoch_rsi_chart(df, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    
    df_2m = df.copy()
    
    # 1. 轉換為 2 個月 (2ME) 週期 K 線
    df_2m['open_time'] = pd.to_datetime(df_2m['open_time'])
    df_2m = df_2m.set_index('open_time')
    # 依照你的 pandas 版本，若 '2ME' 報錯可改為 '2M'
    df_2m = df_2m.resample('2ME').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # 2. 計算 RSI (使用 Wilder's Smoothing)
    delta = df_2m['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_2m['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. 計算 Stoch RSI
    min_rsi = df_2m['rsi'].rolling(window=stoch_period).min()
    max_rsi = df_2m['rsi'].rolling(window=stoch_period).max()
    df_2m['stoch_rsi'] = 100 * (df_2m['rsi'] - min_rsi) / (max_rsi - min_rsi)
    
    # 4. 平滑處理 (K 與 D)
    df_2m['K'] = df_2m['stoch_rsi'].rolling(window=k_period).mean()
    df_2m['D'] = df_2m['K'].rolling(window=d_period).mean()
    
    df_2m = df_2m.reset_index()
    
    # 取 2014 之後的數據以利圖表呈現
    plot_data = df_2m[df_2m['open_time'] >= "2014-01-01"].copy()

    # 5. 偵測黃金交叉與死亡交叉
    # 買點：在超賣區 (<30) 金叉
    cross_buy = (plot_data['K'] > plot_data['D']) & \
                (plot_data['K'].shift(1) <= plot_data['D'].shift(1)) & \
                (plot_data['D'] < 30)
                
    # 賣點：在超買區 (>70) 死叉
    cross_sell = (plot_data['K'] < plot_data['D']) & \
                 (plot_data['K'].shift(1) >= plot_data['D'].shift(1)) & \
                 (plot_data['D'] > 70)

    # 6. 繪圖
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4], 
        subplot_titles=("BTC Price (雙月 K 線)", "2-Month Stoch RSI")
    )

    # --- 上圖: BTC 雙月價格 ---
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['close'],
        mode='lines', name='Price',
        line=dict(color='#F7931A', width=2)
    ), row=1, col=1)

    # 標記大底 (金叉 < 30) 附帶文字與垂直線
    if cross_buy.any():
        buy_pts = plot_data[cross_buy]
        fig.add_trace(go.Scatter(
            x=buy_pts['open_time'], 
            y=buy_pts['close'] * 0.8,
            mode='markers+text',
            name='大底金叉 (<30)',
            marker=dict(symbol='triangle-up', size=18, color='#00e676', line=dict(width=2, color='white')),
            text=["⭐大底金叉"] * len(buy_pts),
            textposition="bottom center",
            textfont=dict(color='#00e676', size=14, weight='bold')
        ), row=1, col=1)
        
        for date in buy_pts['open_time']:
            fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="rgba(0, 230, 118, 0.5)")

    # 標記大頂 (死叉 > 70) 附帶文字與垂直線
    if cross_sell.any():
        sell_pts = plot_data[cross_sell]
        fig.add_trace(go.Scatter(
            x=sell_pts['open_time'], 
            y=sell_pts['close'] * 1.2, 
            mode='markers+text', 
            name='大頂死叉 (>70)',
            marker=dict(symbol='triangle-down', size=18, color='#ff1744', line=dict(width=2, color='white')),
            text=["💀大頂死叉"] * len(sell_pts),
            textposition="top center",
            textfont=dict(color='#ff1744', size=14, weight='bold')
        ), row=1, col=1)
        
        for date in sell_pts['open_time']:
            fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="rgba(255, 23, 68, 0.5)")

    # --- 下圖: Stoch RSI ---
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['K'],
        mode='lines', name='%K (快線)',
        line=dict(color='#00e5ff', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['D'],
        mode='lines', name='%D (慢線)',
        line=dict(color='#ff9100', width=2, dash='dot')
    ), row=2, col=1)

    # 背景區間標示
    fig.add_hrect(y0=0, y1=30, fillcolor="#00e676", opacity=0.15, line_width=0, row=2, col=1, annotation_text="歷史築底區 (<30)", annotation_position="top left")
    fig.add_hrect(y0=70, y1=100, fillcolor="#ff1744", opacity=0.15, line_width=0, row=2, col=1, annotation_text="歷史逃頂區 (>70)", annotation_position="bottom left")
    fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff1744", row=2, col=1)

    # 當前狀態
    curr_k = plot_data.iloc[-1]['K']
    curr_d = plot_data.iloc[-1]['D']
    
    if curr_k < 30 or curr_d < 30:
        status = "💎 指標已進入超賣區 (歷史大底醞釀中)"
        color = "#00e676"
    elif curr_k > 70 or curr_d > 70:
        status = "🔥 指標已進入超買區 (風險極高)"
        color = "#ff1744"
    else:
        status = "⚖️ 中性區間 (趨勢延續中)"
        color = "#ffffff"

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.95,
        text=f"<b>Stoch RSI %K: {curr_k:.1f}</b><br><span style='color:{color}'>{status}</span>",
        showarrow=False, bgcolor="rgba(0,0,0,0.8)", bordercolor=color, borderwidth=1,
        font=dict(size=14, color="white"), align="left"
    )

    fig.update_layout(
        template="plotly_dark", height=650, hovermode='x unified',
        yaxis=dict(title="Price (Log)", type='log'),
        yaxis2=dict(title="Stoch RSI (0-100)", range=[-5, 105]),
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig, curr_k, curr_d

# ============================================================
# MFI 資金流量 (Smart Money Flow) - 修復價格顯示
# ============================================================
def get_mfi_divergence_chart(df, period=14):
    data = df.copy()
    
    # 1. 計算 MFI
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3
    data['rmf'] = data['tp'] * data['volume']
    data['pmf'] = np.where(data['tp'] > data['tp'].shift(1), data['tp'] * data['volume'], 0)
    data['nmf'] = np.where(data['tp'] < data['tp'].shift(1), data['tp'] * data['volume'], 0)
    mfr = data['pmf'].rolling(window=period).sum() / data['nmf'].rolling(window=period).sum()
    data['mfi'] = 100 - (100 / (1 + mfr))
    
    # 取 2018 之後
    plot_data = data[data['open_time'] >= "2018-01-01"].copy()

    # 2. 繪圖設定
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.6, 0.4], 
        subplot_titles=("BTC Price Action", "MFI Money Flow (資金流向溫度計)")
    )

    # --- 上圖: 價格 (修改重點：改顏色、加粗) ---
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['close'],
        mode='lines', name='Price',
        line=dict(color='#F7931A', width=2) # <--- 改成亮橙色，寬度設為 2
    ), row=1, col=1)

    # --- 下圖: MFI 區域圖 ---
    # 1. 繪製主線
    fig.add_trace(go.Scatter(
        x=plot_data['open_time'], y=plot_data['mfi'],
        mode='lines', name='MFI Flow',
        line=dict(color='#2979ff', width=1),
        fill='tozeroy', 
        fillcolor='rgba(41, 121, 255, 0.1)' 
    ), row=2, col=1)

    # 2. 標記極值點
    overbought = plot_data[plot_data['mfi'] >= 80]
    oversold = plot_data[plot_data['mfi'] <= 20]

    fig.add_trace(go.Scatter(
        x=overbought['open_time'], y=overbought['mfi'],
        mode='markers', name='資金過熱 (Retail FOMO)',
        marker=dict(color='#ff1744', size=5),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=oversold['open_time'], y=oversold['mfi'],
        mode='markers', name='資金冰點 (Smart Money Buy)',
        marker=dict(color='#00e676', size=5),
    ), row=2, col=1)

    # 3. 繪製參考線
    fig.add_hline(y=80, line_dash="dot", line_color="#ff1744", row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#00e676", row=2, col=1)
    
    # 4. 當前狀態解讀
    last_mfi = plot_data.iloc[-1]['mfi']
    
    if last_mfi > 80:
        status = "🔥 警告：買力耗盡 (Buyer Exhaustion)"
        desc = "散戶資金已全數進場，後續缺乏推升力道。"
        s_color = "#ff1744"
    elif last_mfi < 20:
        status = "💎 機會：賣壓衰竭 (Seller Exhaustion)"
        desc = "市場無人想賣，聰明錢開始吸籌。"
        s_color = "#00e676"
    else:
        status = "⚖️ 資金流動正常 (Neutral Flow)"
        desc = "多空力量均衡，跟隨趨勢。"
        s_color = "#ffffff"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.95,
        text=f"<b>Current MFI: {last_mfi:.1f}</b><br>{status}<br><span style='font-size:10px;color:gray'>{desc}</span>",
        showarrow=False,
        bgcolor="rgba(0,0,0,0.8)", bordercolor=s_color, borderwidth=1,
        font=dict(size=14, color=s_color),
        align="left"
    )

    fig.update_layout(
        template="plotly_dark", height=550, hovermode='x unified',
        yaxis=dict(title="Price", type='log'),
        yaxis2=dict(title="MFI Index", range=[0, 100]),
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig, last_mfi
# ============================================================
# Mayer Multiple (梅耶倍數) - 宏觀估值
# ============================================================
def get_mayer_multiple_chart(df):
    data = df.copy()
    
    # 1. 計算 200 日均線
    data['sma200'] = data['close'].rolling(window=200).mean()
    
    # 2. 計算 Mayer Multiple
    data['mayer'] = data['close'] / data['sma200']
    
    # 取 2012 之後 (避開早期極端值)
    plot_data = data[data['open_time'] >= "2012-01-01"].copy()
    
    # 3. 繪圖
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3], 
        subplot_titles=("BTC Price & Bands", "Mayer Multiple Ratio")
    )

    # --- 上圖: 價格 + 帶狀 ---
    fig.add_trace(go.Scatter(x=plot_data['open_time'], y=plot_data['close'], mode='lines', name='Price', line=dict(color='#F7931A')), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_data['open_time'], y=plot_data['sma200'], mode='lines', name='200 DMA', line=dict(color='yellow')), row=1, col=1)
    
    # 畫出 2.4 倍 (泡沫線) 和 0.6 倍 (抄底線)
    fig.add_trace(go.Scatter(x=plot_data['open_time'], y=plot_data['sma200']*2.4, mode='lines', name='Bubble (2.4x)', line=dict(color='red', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_data['open_time'], y=plot_data['sma200']*0.6, mode='lines', name='Buy (0.6x)', line=dict(color='green', dash='dot')), row=1, col=1)

    # --- 下圖: Mayer Multiple 指標 ---
    # 根據數值變色
    colors = np.where(plot_data['mayer'] > 2.4, '#ff1744', 
             np.where(plot_data['mayer'] < 0.6, '#00e676', '#2979ff'))
    
    fig.add_trace(go.Bar(x=plot_data['open_time'], y=plot_data['mayer'], name='Mayer Multiple', marker_color=colors), row=2, col=1)

    # 參考線
    fig.add_hline(y=1.0, line_color="white", line_width=1, row=2, col=1)
    fig.add_hline(y=2.4, line_color="red", line_dash="dash", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=0.6, line_color="green", line_dash="dash", row=2, col=1, annotation_text="Undervalued")

    fig.update_layout(template="plotly_dark", height=600, hovermode='x unified', yaxis=dict(type='log'), title="Mayer Multiple (梅耶倍數)")
    
    return fig, plot_data.iloc[-1]['mayer']
# ============================================================
# 宏觀指標：銅金比 (Copper/Gold Ratio) - 穩健修復版 (Log通道)
# ============================================================
def get_copper_gold_ratio_chart(lookback_years=10):
    import yfinance as yf
    from sklearn.linear_model import LinearRegression

    # 1. 抓取數據 (銅 & 黃金)
    start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    
    try:
        # 分開抓取以避免 MultiIndex 結構混亂
        copper = yf.download("HG=F", start=start_date, progress=False)
        gold = yf.download("GC=F", start=start_date, progress=False)
        
        # 檢查數據
        if copper.empty or gold.empty:
            st.error("❌ 無法抓取銅或黃金數據。")
            return go.Figure(), 0

        # 通用函數：從 yfinance 結果中提取 Close 欄位
        def get_close_col(df):
            # 處理各種可能的欄位名稱
            if 'Close' in df.columns: return df['Close']
            if 'close' in df.columns: return df['close']
            # 如果是 MultiIndex，嘗試取第一層的第一欄
            if isinstance(df.columns, pd.MultiIndex):
                try: return df.xs('Close', axis=1, level=0).iloc[:, 0]
                except: return df.iloc[:, 0]
            return df.iloc[:, 0]

        df_macro = pd.DataFrame()
        df_macro['Copper'] = get_close_col(copper)
        df_macro['Gold'] = get_close_col(gold)
        
        df_macro = df_macro.dropna()
        
        if df_macro.empty:
            return go.Figure(), 0

        # 2. 計算銅金比
        df_macro['Ratio'] = df_macro['Copper'] / df_macro['Gold']
        df_macro = df_macro.reset_index()
        
        # 統一日期欄位
        date_col = next((c for c in df_macro.columns if c.lower() in ['date', 'datetime', 'index']), None)
        if date_col: df_macro = df_macro.rename(columns={date_col: 'date'})

        # 3. 計算【Log 線性回歸通道】
        # 這是為了讓它在 Log 圖表上看起來是直的平行通道
        df_macro['Time_Index'] = np.arange(len(df_macro))
        X = df_macro[['Time_Index']]
        
        # 對 Ratio 取 Log
        y_log = np.log(df_macro['Ratio'].values.reshape(-1, 1))

        # 訓練回歸
        reg = LinearRegression().fit(X, y_log)
        log_pred = reg.predict(X).flatten()

        # 計算標準差
        std_dev_log = np.std(y_log - log_pred.reshape(-1, 1))
        
        # 還原回真實數值 (exp)
        df_macro['Reg_Line'] = np.exp(log_pred)
        df_macro['Upper_2std'] = np.exp(log_pred + (2.0 * std_dev_log))
        df_macro['Lower_2std'] = np.exp(log_pred - (2.0 * std_dev_log))
        
        # 4. 繪圖
        fig = go.Figure()

        # (A) 銅金比線
        fig.add_trace(go.Scatter(
            x=df_macro['date'], y=df_macro['Ratio'],
            mode='lines', name='Copper/Gold Ratio',
            line=dict(color='#00E5FF', width=1.5)
        ))

        # (B) 回歸通道
        fig.add_trace(go.Scatter(
            x=df_macro['date'], y=df_macro['Reg_Line'],
            mode='lines', name='Mean',
            line=dict(color='gray', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=df_macro['date'], y=df_macro['Upper_2std'],
            mode='lines', name='Risk-On Top',
            line=dict(color='#ff5252', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=df_macro['date'], y=df_macro['Lower_2std'],
            mode='lines', name='Risk-Off Bottom',
            line=dict(color='#00e676', width=1),
            fill='tonexty', fillcolor='rgba(0, 230, 118, 0.05)'
        ))

        # 5. 狀態解讀
        curr_ratio = df_macro.iloc[-1]['Ratio']
        
        curr_log = np.log(curr_ratio)
        upper_log = np.log(df_macro.iloc[-1]['Upper_2std'])
        lower_log = np.log(df_macro.iloc[-1]['Lower_2std'])
        
        channel_pos = (curr_log - lower_log) / (upper_log - lower_log)
        
        if channel_pos < 0.15:
            status = "💎 觸底 (Macro Bottom)"; color = "#00e676"
        elif channel_pos > 0.85:
            status = "🔥 過熱 (Macro Top)"; color = "#ff5252"
        else:
            status = "⚖️ 中性震盪"; color = "#ffffff"

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.95,
            text=f"<b>Copper/Gold: {curr_ratio:.5f}</b><br>{status}<br>(Pos: {channel_pos:.0%})",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.8)", bordercolor=color, borderwidth=1,
            font=dict(size=14, color=color),
            align="left"
        )

        fig.update_layout(
            title="Copper/Gold Ratio (Log Regression Channel)",
            template="plotly_dark",
            height=500,
            hovermode='x unified',
            yaxis=dict(title="Ratio (Log)", type='log', autorange=True), # 開啟 Log 座標
            legend=dict(orientation="h", y=1.02)
        )
        
        return fig, curr_ratio

    except Exception as e:
        print(f"Copper/Gold Error: {e}")
        return go.Figure(), 0
# ============================================================
# 宏觀指標：Fed Net Liquidity (真實數據 + 單位修正版)
# ============================================================
def get_global_liquidity_chart(lookback_years=5):
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta
    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st
    
    start_date = datetime.now() - timedelta(days=lookback_years*365)
    
    try:
        # 1. 從 FRED 抓取數據
        # WALCL = Fed Total Assets (Millions)
        # WTREGEN = TGA (Billions)
        # RRPONTSYD = Reverse Repo (Billions)
        tickers = {
            'WALCL': 'Assets',
            'WTREGEN': 'TGA', 
            'RRPONTSYD': 'RRP'
        }
        
        try:
            # 【核心修正】：棄用 pandas_datareader，改用 requests + 偽裝標頭
            import requests
            import io
            
            # 戴上假面具，假裝我們是正常的 Chrome 瀏覽器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            dfs = []
            for ticker, name in tickers.items():
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
                
                # 發送請求獲取資料
                resp = requests.get(url, headers=headers, timeout=10)
                
                if resp.status_code != 200:
                    raise Exception(f"被擋下了，HTTP 狀態碼: {resp.status_code}")
                    
                # FRED 的 CSV 遇到假日無資料時會填入 '.'，利用 na_values 轉成 NaN
                temp_df = pd.read_csv(io.StringIO(resp.text), index_col='DATE', parse_dates=True, na_values='.')
                temp_df.rename(columns={ticker: name}, inplace=True)
                dfs.append(temp_df)
            
            # 合併數據並篩選日期
            df_fred = dfs[0].join(dfs[1:], how='outer').sort_index()
            # 確保索引具有時區意識以便後續合併
            df_fred.index = pd.to_datetime(df_fred.index)
            df_fred = df_fred[df_fred.index >= start_date]
            
        except Exception as e:
            st.error(f"連線 FRED 失敗: {e}")
            return go.Figure(), 0
            
        # -------------------------------------------------------
        # 【關鍵修正】單位統一：全部轉為 Billions (十億美元)
        # -------------------------------------------------------
        # Fed Assets 原始數據通常是 Millions，所以要除以 1000
        # 判斷邏輯：如果數值大於 10,000，代表它是 Millions，需要轉換
        # 注意：此處需用 .dropna() 防止 NaN 影響判斷
        valid_assets = df_fred['Assets'].dropna()
        if not valid_assets.empty and valid_assets.iloc[-1] > 10000:
            df_fred['Assets'] = df_fred['Assets'] / 1000
            
        valid_tga = df_fred['TGA'].dropna()
        if not valid_tga.empty and valid_tga.iloc[-1] > 10000: 
            df_fred['TGA'] /= 1000
            
        valid_rrp = df_fred['RRP'].dropna()
        if not valid_rrp.empty and valid_rrp.iloc[-1] > 10000: 
            df_fred['RRP'] /= 1000
            
        # -------------------------------------------------------
        
        # 頻率對齊：將週數據平滑為日數據
        df_fred = df_fred.resample('D').mean().interpolate(method='linear')
        df_fred = df_fred.ffill().dropna()
        
        # 計算淨流動性 (Billions)
        df_fred['Net_Liquidity'] = df_fred['Assets'] - df_fred['TGA'] - df_fred['RRP']
        
        # 2. 抓取 BTC
        btc_data = yf.download("BTC-USD", start=start_date, progress=False)
        
        # 防呆提取 Close
        if isinstance(btc_data, pd.DataFrame):
            if 'Close' in btc_data.columns: btc_series = btc_data['Close']
            elif 'close' in btc_data.columns: btc_series = btc_data['close']
            else: btc_series = btc_data.iloc[:, 0]
        else: btc_series = btc_data
        
        if isinstance(btc_series, pd.DataFrame): btc_series = btc_series.iloc[:, 0]
        btc_series.name = 'BTC'
        
        # 3. 合併
        df_fred.index = df_fred.index.tz_localize(None)
        btc_series.index = btc_series.index.tz_localize(None)
        
        df_combined = pd.DataFrame(index=btc_series.index)
        df_combined['BTC'] = btc_series
        df_combined = df_combined.join(df_fred[['Net_Liquidity']], how='left')
        
        # 再次平滑填補 (因為 BTC 週末有價，但 FRED 週末無數據)
        df_combined['Net_Liquidity'] = df_combined['Net_Liquidity'].interpolate(method='linear')
        df_combined = df_combined.dropna()

        # 4. 繪圖
        fig = go.Figure()

        # (A) 淨流動性 (左軸)
        fig.add_trace(go.Scatter(
            x=df_combined.index, y=df_combined['Net_Liquidity'],
            mode='lines', name='Fed Net Liquidity (Billions)',
            line=dict(color='#2962FF', width=2),
            fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'
        ))

        # (B) BTC (右軸)
        fig.add_trace(go.Scatter(
            x=df_combined.index, y=df_combined['BTC'],
            mode='lines', name='Bitcoin Price',
            line=dict(color='#FF9100', width=2),
            yaxis='y2'
        ))

        curr_liq = df_combined['Net_Liquidity'].iloc[-1]
        
        # 確保數值正常 (避免顯示 0 或 NaN)
        if pd.isna(curr_liq) or curr_liq < 0:
             curr_liq = df_combined['Net_Liquidity'].iloc[-2] 

        # 狀態計算
        change = 0
        if len(df_combined) > 30:
            change = curr_liq - df_combined['Net_Liquidity'].iloc[-30]
            
        status_text = "擴張 (Easing)" if change > 0 else "緊縮 (Tightening)"
        color = "#00e676" if change > 0 else "#ff5252"

        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.95,
            text=f"<b>Fed Net Liquidity: ${curr_liq:,.0f} B</b><br><span style='color:{color}'>{status_text} (MoM: {change:+.0f}B)</span>",
            showarrow=False, bgcolor="rgba(0,0,0,0.8)", bordercolor=color, borderwidth=1,
            font=dict(size=14, color="white"), align="left"
        )

        fig.update_layout(
            title="Fed Net Liquidity vs Bitcoin (真實美元流動性)",
            template="plotly_dark", height=550, hovermode='x unified',
            legend=dict(orientation="h", y=1.05),
            yaxis=dict(title="Net Liquidity (Billions USD)", showgrid=True, gridcolor='rgba(255,255,255,0.1)', autorange=True),
            yaxis2=dict(title="BTC Price (Log)", overlaying='y', side='right', showgrid=False, type='log')
        )
        
        return fig, curr_liq

    except Exception as e:
        print(f"FRED Error: {e}")
        st.error(f"無法獲取數據: {e}")
        return go.Figure(), 0
# ============================================================
# BTC vs S&P 500 季度脫鉤分析 (90-Day Decoupling)
# ============================================================
def get_btc_spx_decoupling_chart(df_btc, lookback_years=8, window=90):
    import yfinance as yf
    
    # 1. 抓取標普 500 數據
    start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
    spx_data = yf.download("^GSPC", start=start_date, progress=False)
    
    if spx_data.empty:
        return None, None
    
    # 2. 數據對齊 (BTC 24/7 vs SPX 5天交易)
    spx_close = spx_data['Close'].copy()
    if isinstance(spx_close, pd.DataFrame): spx_close = spx_close.iloc[:, 0]
    
    df_macro = pd.DataFrame(index=df_btc['open_time'])
    df_macro['BTC'] = df_btc.set_index('open_time')['close']
    df_macro['SPX'] = spx_close
    df_macro['SPX'] = df_macro['SPX'].ffill() # 填補美股週末空窗
    df_macro = df_macro.dropna()

    # 3. 計算 90 天滾動相關性
    df_macro['corr'] = df_macro['BTC'].rolling(window).corr(df_macro['SPX'])

    # 4. 偵測脫鉤區間與統計
    # 定義：相關係數 < 0.2 視為脫鉤 (Decoupled)
    df_macro['is_decoupled'] = df_macro['corr'] < 0.2
    
    # [新增這段代碼]：計算「當前」連續脫鉤天數
    current_streak_count = 0
    # 從最後一筆資料往回數，直到遇到第一個「未脫鉤」為止
    for val in df_macro['is_decoupled'].iloc[::-1]:
        if val:
            current_streak_count += 1
        else:
            break

    # 計算連續脫鉤天數
    streaks = []
    current_streak = 0
    for val in df_macro['is_decoupled']:
        if val:
            current_streak += 1
        else:
            if current_streak > 0: streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0: streaks.append(current_streak)
    
    avg_decouple = np.mean(streaks) if streaks else 0
    max_decouple = np.max(streaks) if streaks else 0

    # 5. 繪圖
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4], 
        subplot_titles=("價格走勢對比 (Normalized to Start)", f"{window}日滾動相關性 (季度視角)")
    )

    # (A) 歸一化價格線
    # 讓兩者起點相同，方便看誰漲得猛
    btc_norm = df_macro['BTC'] / df_macro['BTC'].iloc[0]
    spx_norm = df_macro['SPX'] / df_macro['SPX'].iloc[0]
    
    fig.add_trace(go.Scatter(x=df_macro.index, y=btc_norm, name='BTC (比特幣)', line=dict(color='#F7931A', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_macro.index, y=spx_norm, name='S&P 500 (美股標普)', line=dict(color='#00d3ff', width=1, dash='dot')), row=1, col=1)

    # (B) 相關性柱狀圖
    # 綠色 = 脫鉤 (各走各的)；紅色 = 耦合 (同漲同跌)
    colors = np.where(df_macro['corr'] < 0.2, '#00e676', '#ff5252')
    
    fig.add_trace(go.Bar(
        x=df_macro.index, y=df_macro['corr'],
        name='Correlation (r)',
        marker_color=colors,
        opacity=0.7
    ), row=2, col=1)

    # 參考線
    fig.add_hline(y=0.2, line_dash="dash", line_color="#00e676", row=2, col=1, annotation_text="脫鉤門檻")
    fig.add_hline(y=0.7, line_dash="dash", line_color="#ff5252", row=2, col=1, annotation_text="高度正相關")
    # 將 width 改為 line_width
    fig.add_hline(y=0, line_color="white", line_width=1, row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=650, hovermode='x unified',
        yaxis=dict(title="Relative Scale", type='log'),
        yaxis2=dict(title="Correlation Score", range=[-1, 1]),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.05)
    )
    
    stats = {
        "avg": avg_decouple,
        "max": max_decouple,
        "current": df_macro['corr'].iloc[-1],
        "current_streak": current_streak_count,  # <--- 修改點 1：加入這一行
        "is_decoupled": df_macro['is_decoupled'].iloc[-1],
        "window": window
    }
    
    return fig, stats
# ============================================================
# 鏈上指標：CDD (Coin Days Destroyed) - 老幣甦醒偵測
# ============================================================
def fetch_blockchain_cdd():
    # Blockchain.com 免費 API 提供 CDD
    url = "https://api.blockchain.info/charts/bitcoin-days-destroyed?timespan=5years&format=json&cors=true"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['date'] = pd.to_datetime(df['x'], unit='s')
            df = df.rename(columns={'y': 'cdd'})
            return df[['date', 'cdd']]
    except Exception as e:
        print(f"Error fetching CDD: {e}")
    return pd.DataFrame()

def get_cdd_chart(df_cdd, df_price):
    if df_cdd.empty:
        return None

    # 1. 數據處理
    # CDD 數據波動極大，必須做平滑處理 (例如 7日或 30日移動平均)
    df_cdd = df_cdd.sort_values('date')
    df_cdd['cdd_ma'] = df_cdd['cdd'].rolling(window=30).mean() # 30日平滑，看趨勢
    
    # 為了對比，我們需要把 BTC 價格也併進來
    df_merge = pd.merge_asof(df_cdd, df_price[['open_time', 'close']], 
                             left_on='date', right_on='open_time', 
                             direction='nearest')

    # 取 2020 之後
    plot_data = df_merge[df_merge['date'] >= "2020-01-01"].copy()

    # 2. 繪圖
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.6, 0.4], 
        subplot_titles=("Price Action", "Coin Days Destroyed (30D MA)")
    )

    # 上圖：價格
    fig.add_trace(go.Scatter(
        x=plot_data['date'], y=plot_data['close'],
        mode='lines', name='Price',
        line=dict(color='#F7931A', width=1.5)
    ), row=1, col=1)

    # 下圖：CDD
    # 這裡我們用「柱狀圖」還是「區域圖」？區域圖比較能看清趨勢
    fig.add_trace(go.Scatter(
        x=plot_data['date'], y=plot_data['cdd_ma'],
        mode='lines', name='CDD (老幣異動)',
        line=dict(color='#00e5ff', width=1),
        fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)'
    ), row=2, col=1)

    # 標記「異常活躍」的水平線 (例如歷史高位)
    # 這數值是經驗值，通常 CDD MA30 超過 1500萬-2000萬 代表老幣顯著移動
    threshold = plot_data['cdd_ma'].quantile(0.90) # 取歷史前 10% 作為警戒線
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ff1744", row=2, col=1, annotation_text="高換手警戒區")

    fig.update_layout(
        template="plotly_dark", height=600, hovermode='x unified',
        yaxis=dict(title="Price", type='log'),
        yaxis2=dict(title="CDD (Days)"),
        legend=dict(orientation="h", y=1.02)
    )
    
    return fig, plot_data.iloc[-1]['cdd_ma']
# ============================================================
# 3. 風險計算邏輯
# ============================================================

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
        df_youtube_activity['date'] = pd.to_datetime(df_youtube_activity['date'])
        df_youtube_activity = df_youtube_activity.sort_values('date')
        
        videos_norm = (df_youtube_activity['video_count'] - df_youtube_activity['video_count'].min()) / \
                      (df_youtube_activity['video_count'].max() - df_youtube_activity['video_count'].min() + 1)
        log_views = np.log10(df_youtube_activity['avg_views'] + 1)
        views_norm = (log_views - log_views.min()) / (log_views.max() - log_views.min() + 0.001)
        heat_norm = df_youtube_activity['high_view_ratio']
        
        df_youtube_activity['composite_score'] = 0.3 * videos_norm + 0.4 * views_norm + 0.3 * heat_norm
        
        temp_df = df[['open_time']].copy()
        temp_df['date'] = temp_df['open_time']
        merged_yt = pd.merge_asof(temp_df, df_youtube_activity[['date', 'composite_score']], 
                                  on='date', direction='backward')
        
        df['youtube_val'] = merged_yt['composite_score'].values
        df['youtube_val'] = df['youtube_val'].ffill().bfill().fillna(0)
        
        yt_risk = df['youtube_val'].rank(pct=True).rolling(7, min_periods=1).mean().clip(0, 1)
        social_risk_components.append(('youtube', yt_risk, YOUTUBE_WEIGHT))
    else:
        df['youtube_val'] = 0

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


@st.cache_data(ttl=3600)
def load_data_and_compute():
    # =========================================================
    # 0. 建立 Google Sheets 連線
    # =========================================================
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    
    # 定義讀取 Sheet 的小幫手 (包含跳過浮水印邏輯)
    def read_sheet_to_df(client, sheet_url, worksheet_name):
        try:
            sh = client.open_by_url(sheet_url)
            ws = sh.worksheet(worksheet_name)
            raw_data = ws.get_all_values()
            
            if not raw_data:
                return pd.DataFrame()

            # 🕵️‍♂️ 邏輯移植：判斷第一列是否為垃圾資訊 (網址/浮水印)
            if len(raw_data) > 1 and ("http" in str(raw_data[0][0]) or "Crypto" in str(raw_data[0][0])):
                headers = raw_data[1] # 跳過第一列，用第二列當標題
                rows = raw_data[2:]
            else:
                headers = raw_data[0]
                rows = raw_data[1:]
            
            return pd.DataFrame(rows, columns=headers)
        except Exception as e:
            print(f"⚠️ 讀取 {worksheet_name} 失敗: {e}")
            return pd.DataFrame()

    # 初始化 gspread
    try:
        # 1. 從 secrets 讀取設定，並轉成 Python 字典
        # (原本你是讀 service_account.json，現在我們改讀這個字典，內容其實一模一樣)
        creds_dict = dict(st.secrets["gsheets"])
        
        # 2. 【超級關鍵】修正 Private Key 的換行符號
        # 這行程式碼會自動把 secrets 裡的文字 "\n" 轉成真正的換行，解決 401 錯誤
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

        # 3. 告訴 gspread：請讀這個字典 (from_json_keyfile_dict)
        # 這樣就不需要 service_account.json 檔案了！
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        
        # 4. 建立連線
        client = gspread.authorize(creds)
        
        # 5. 讀取網址
        sheet_url = st.secrets["gsheets"]["spreadsheet"]
        
        print("✅ gspread 連線成功 (使用 Secrets)！")
        
    except Exception as e:
        st.error(f"❌ gspread 連線失敗: {e}")
        return pd.DataFrame()

    # =========================================================
    # 1. 抓取 Binance 數據 (2017-08-17 ~ Now)
    # =========================================================
    symbol = "BTCUSDT"
    df_binance = fetch_binance_klines(symbol=symbol)
    if not df_binance.empty:
        # 防止 Binance 數據本身有重複欄位
        df_binance = df_binance.loc[:, ~df_binance.columns.duplicated()]

    # =========================================================
    # 2. 讀取 Google Sheet 歷史數據 (取代本地 CSV)
    # =========================================================
    df_history = pd.DataFrame()
    
    # 讀取 price_data 分頁
    df_raw = read_sheet_to_df(client, sheet_url, "price_data")
    
    if not df_raw.empty:
        try:
            # --- 以下完全移植您原本的 CSV 清洗邏輯 ---
            
            # 1. 清洗欄位名稱 (轉小寫、去空白)
            df_raw.columns = [str(c).strip().lower() for c in df_raw.columns]
            
            # 2. 欄位對應 (Mapping)
            col_map = {}
            for col in df_raw.columns:
                if 'date' in col: col_map[col] = 'open_time'
                elif 'unix' in col: continue 
                elif 'close' in col: col_map[col] = 'close'
                elif 'volume' in col: col_map[col] = 'volume'
                elif 'open' in col: col_map[col] = 'open'
                elif 'high' in col: col_map[col] = 'high'
                elif 'low' in col: col_map[col] = 'low'

            df_raw = df_raw.rename(columns=col_map)
            
            # 3. 【關鍵修復】移除重複的欄位名稱
            df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
            
            # 4. 確保必要的欄位存在並處理數據
            if 'open_time' in df_raw.columns:
                df_raw['open_time'] = pd.to_datetime(df_raw['open_time'])
                df_raw = df_raw.sort_values('open_time')
                
                # 數值轉換 (Sheet 讀下來是字串，必須轉數字)
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    if c in df_raw.columns:
                        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

                # 只取 Binance 之前的數據
                df_history = df_raw[df_raw['open_time'] < "2017-08-17"].reset_index(drop=True)
                
                # 補齊缺少的 OHLC (如果只有 Close)
                if 'close' in df_history.columns:
                    for c in ['open', 'high', 'low']:
                        if c not in df_history.columns: df_history[c] = df_history['close']
                
                # 如果沒有 volume，補 1000
                if 'volume' not in df_history.columns: 
                    df_history['volume'] = 1000 
                
                # 只保留需要的欄位，進一步淨化
                required_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
                # 確保欄位都存在才選取
                existing_cols = [c for c in required_cols if c in df_history.columns]
                df_history = df_history[existing_cols]
                
                print(f"✅ 成功從 Sheet 讀取並清洗歷史數據！共 {len(df_history)} 筆。")

        except Exception as e:
            print(f"Sheet cleaning logic failed: {e}")
            pass

    # =========================================================
    # 3. 讀取 Wiki 和 YT 數據 (從 Sheet)
    # =========================================================
    df_wiki_sheet = read_sheet_to_df(client, sheet_url, "wiki_data")
    if not df_wiki_sheet.empty:
        df_wiki_sheet.columns = [str(c).strip().lower() for c in df_wiki_sheet.columns]
        if 'date_wiki' in df_wiki_sheet.columns:
            df_wiki_sheet['date_wiki'] = pd.to_datetime(df_wiki_sheet['date_wiki'])
        # Wiki 也是，除了日期以外通通轉數字
        for col in df_wiki_sheet.columns:
            if col != 'date_wiki':
                df_wiki_sheet[col] = pd.to_numeric(df_wiki_sheet[col], errors='coerce').fillna(0)

    # 👇👇👇 重點修改這裡 (地毯式轉型) 👇👇👇
    df_yt_sheet = read_sheet_to_df(client, sheet_url, "yt_data")
    if not df_yt_sheet.empty:
        df_yt_sheet.columns = [str(c).strip().lower() for c in df_yt_sheet.columns]
        
        # 1. 處理日期
        if 'date' in df_yt_sheet.columns:
            df_yt_sheet['date'] = pd.to_datetime(df_yt_sheet['date'])
        
        # 2. 【強力修復】不管欄位叫什麼名字，只要不是日期，全部強制轉成數字
        # 這會一次解決 avg_views, subscriber_count, heat_score 等所有欄位的問題
        for col in df_yt_sheet.columns:
            if col != 'date':
                # errors='coerce' 會把無法轉數字的變成 NaN，然後 fillna(0) 補成 0
                df_yt_sheet[col] = pd.to_numeric(df_yt_sheet[col], errors='coerce').fillna(0)
    # =========================================================
    # 4. 早期模擬數據 (2010-2014) - 保持原本邏輯
    # =========================================================
    real_points = [
        ("2010-07-17", 0.05), ("2011-02-09", 1.00), ("2011-06-08", 31.00),
        ("2011-11-14", 2.00), ("2012-02-20", 5.00), ("2012-08-17", 10.00),
        ("2013-04-09", 230.00), ("2013-07-06", 66.00), ("2013-11-30", 1150.00),
        ("2014-04-10", 360.00), ("2014-09-16", 450.00)
    ]
    
    data_rows = []
    for i in range(len(real_points)-1):
        d1 = pd.to_datetime(real_points[i][0])
        p1 = real_points[i][1]
        d2 = pd.to_datetime(real_points[i+1][0])
        p2 = real_points[i+1][1]
        
        days = (d2 - d1).days
        if days <= 0: continue
        
        prices = np.logspace(np.log10(p1), np.log10(p2), days, endpoint=False)
        noise = np.random.normal(0, p1*0.05, days)
        prices = prices + noise
        
        curr = d1
        for p in prices:
            if p <= 0: p = 0.01
            data_rows.append({
                "open_time": curr, "open": p, "high": p, "low": p, "close": p, "volume": p*1000
            })
            curr += timedelta(days=1)
            
    df_early = pd.DataFrame(data_rows)
    df_early = df_early.loc[:, ~df_early.columns.duplicated()]

    # =========================================================
    # 5. 數據合併
    # =========================================================
    df_final_list = []

    # A. 處理早期模擬數據
    if not df_history.empty:
        min_hist_date = df_history['open_time'].min()
        df_early = df_early[df_early['open_time'] < min_hist_date]
    
    if not df_early.empty:
        df_final_list.append(df_early[['open_time', 'open', 'high', 'low', 'close', 'volume']])

    # B. 加入 Sheet 歷史數據 (Bitstamp替代品)
    if not df_history.empty:
        df_final_list.append(df_history[['open_time', 'open', 'high', 'low', 'close', 'volume']])

    # C. 加入 Binance 數據
    if not df_binance.empty:
        df_binance['open_time'] = pd.to_datetime(df_binance['open_time'])
        if not df_history.empty:
            max_hist = df_history['open_time'].max()
            df_binance = df_binance[df_binance['open_time'] > max_hist]
        else:
            max_early = df_early['open_time'].max()
            df_binance = df_binance[df_binance['open_time'] > max_early]
        
        if not df_binance.empty:
            df_final_list.append(df_binance[['open_time', 'open', 'high', 'low', 'close', 'volume']])

    # 合併
    if df_final_list:
        df = pd.concat(df_final_list, ignore_index=True)
        df = df.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'total_risk'])

    # 數值轉換
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    # =========================================================
    # 6. 計算風險
    # =========================================================
    df['date_only'] = df['open_time'].dt.date
    
    # 外部數據 API
    df_fg = fetch_fear_greed_index()
    if not df_fg.empty:
        df_fg['date_only'] = df_fg['date'].dt.date
        df = df.merge(df_fg[['date_only', 'fear_greed']], on='date_only', how='left')
        df['fear_greed'] = df['fear_greed'].interpolate().ffill().bfill()

    # 合併 Wiki (優先使用 Sheet 讀到的)
    if not df_wiki_sheet.empty:
        df_wiki_sheet['date_only'] = df_wiki_sheet['date_wiki'].dt.date
        df = df.merge(df_wiki_sheet[['date_only', 'wiki_views']], on='date_only', how='left')
        df['wiki_views'] = df['wiki_views'].interpolate().ffill().bfill()
    else:
        # 如果 Sheet 沒讀到，才嘗試原本的 API (可選)
        df_wiki = fetch_wikipedia_views_history()
        if not df_wiki.empty:
            df_wiki['date_only'] = df_wiki['date_wiki'].dt.date
            df = df.merge(df_wiki[['date_only', 'wiki_views']], on='date_only', how='left')
            df['wiki_views'] = df['wiki_views'].interpolate().ffill().bfill()
        
    # 其他 API
    df_bc = fetch_blockchain_com_stats()
    df_cg = fetch_coinglass_sentiment()
    df_gn = fetch_google_news_mentions()
    df_obt = fetch_bitcoin_obituaries()
    cmc_rank = fetch_cmc_trending()

    df['date_index'] = pd.to_datetime(df['open_time']).dt.normalize()
    df = df.set_index('date_index', drop=False)

    # 使用 Sheet 讀到的 YT 數據
    yt_source = df_yt_sheet if not df_yt_sheet.empty else load_youtube_activity_history()

    df_full = compute_risk(
        df.copy(), 
        df_blockchain_com=df_bc, df_coinglass=df_cg, df_google_news=df_gn,
        df_obituaries=df_obt, cmc_rank=cmc_rank, df_youtube_activity=yt_source
    )

    cutoff_date = pd.to_datetime("2017-08-17")
    df_recent = df[df['open_time'] >= cutoff_date].copy()
    
    if not df_recent.empty:
        df_recent = compute_risk(
            df_recent,
            df_blockchain_com=df_bc, df_coinglass=df_cg, df_google_news=df_gn,
            df_obituaries=df_obt, cmc_rank=cmc_rank, df_youtube_activity=yt_source
        )
    
    df_final = df_full.copy()
    df_final = df_final.reset_index(drop=True)
    if not df_recent.empty:
        df_recent = df_recent.set_index('open_time')
        df_final = df_final.set_index('open_time')
        cols_to_overwrite = ['price_risk', 'social_risk', 'total_risk']
        df_final.update(df_recent[cols_to_overwrite])
        df_final = df_final.reset_index()
    
    return df_final

# 支援對數座標
def get_risk_chart_figure(df, use_log=False):
    fig = go.Figure()
    
    risk_ranges = [
        (0.0, 0.1, '#0d47a1', '0.0 - 0.1'),
        (0.1, 0.2, '#1976d2', '0.1 - 0.2'),
        (0.2, 0.3, '#2196f3', '0.2 - 0.3'),
        (0.3, 0.4, '#42a5f5', '0.3 - 0.4'),
        (0.4, 0.5, '#4caf50', '0.4 - 0.5'),
        (0.5, 0.6, '#ffeb3b', '0.5 - 0.6'),
        (0.6, 0.7, '#ffc107', '0.6 - 0.7'),
        (0.7, 0.8, '#ff9800', '0.7 - 0.8'),
        (0.8, 0.85, '#ff5722', '0.8 - 0.85'),
        (0.85, 1.01, '#d32f2f', '0.85 - 1.0'),
    ]
    
    fig.add_trace(go.Scatter(
        x=df['open_time'], y=df['close'],
        mode='lines', name='Price',
        line=dict(width=1, color='rgba(255,255,255,0.1)'),
        hoverinfo='skip', showlegend=True
    ))

    dates = df['open_time'].values
    closes = df['close'].values
    risks = df['total_risk'].values
    
    for col in ['fear_greed', 'youtube_val', 'wiki_views', 'blockchain_active', 'coinglass_funding', 'news_count', 'obituary_count', 'cmc_rank']:
        if col not in df.columns: df[col] = 0

    custom_data_array = df[[
        'total_risk', 
        'fear_greed', 
        'youtube_val', 
        'wiki_views', 
        'blockchain_active',
        'coinglass_funding',
        'news_count',
        'obituary_count',
        'cmc_rank'
    ]].values

    hover_template = (
        "<b>日期:</b> %{x|%Y-%m-%d}<br>"
        "<b>價格:</b> $%{y:,.2f}<br>"
        "<b>社交風險指數:</b> %{customdata[0]:.1%}<br>"
        "------------------<br>"
        "<b>F&G:</b> %{customdata[1]:.0f}<br>"
        "<b>YT熱度:</b> %{customdata[2]:.2f}<br>"
        "<b>Wiki:</b> %{customdata[3]:.0f}<br>"
        "<b>地址:</b> %{customdata[4]:.0f}<br>"
        "<b>資金費:</b> %{customdata[5]:.4f}%<br>"
        "<b>新聞:</b> %{customdata[6]:.0f}<br>"
        "<b>死亡:</b> %{customdata[7]:.0f}<br>"
        "<b>CMC:</b> #%{customdata[8]:.0f}<extra></extra>"
    )

    for min_r, max_r, color, label in risk_ranges:
        mask = (risks >= min_r) & (risks < max_r)
        mask_extended = mask.copy()
        if len(mask) > 1:
            mask_shifted = np.roll(mask, 1)
            mask_shifted[0] = False
            mask_extended = mask | mask_shifted
            
        if not np.any(mask_extended):
            continue

        subset_close = closes.copy()
        subset_close[~mask_extended] = np.nan
        
        fig.add_trace(go.Scatter(
            x=dates, y=subset_close,
            mode='lines', name=label,
            line=dict(width=3, color=color),
            connectgaps=False,
            customdata=custom_data_array,
            hovertemplate=hover_template,
            legendgroup=label 
        ))

    layout_args = dict(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        height=600,
        legend=dict(
            title=dict(text="<b>風險等級 (點擊顯示/隱藏)</b>"),
            orientation="v", yanchor="top", y=0.98,
            xanchor="right", x=0.99,
            bgcolor="rgba(0,0,0,0.6)", bordercolor="grey", borderwidth=1,
            font=dict(size=12, color="white"),
            itemsizing='constant', traceorder='normal'
        ),
        hovermode='x unified',
        dragmode='pan',
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
    )
    
    if use_log:
        layout_args['yaxis'] = dict(type='log')

    fig.update_layout(**layout_args)
    return fig

# ============================================================
# 5. DCA 回測功能 (Update with Fees)
# ============================================================

def run_backtest(df, trade_asset, buy_amount, buy_min, buy_max, sell_pct, sell_min, sell_max, start_date, fee_rate=0.001):
    
    # 如果不是 BTC/ETH，需要動態抓取該幣種價格
    price_col = 'asset_price'
    
    df_test = df.copy()
    
    if trade_asset == 'BTC':
        df_test['asset_price'] = df_test['close']
    else:
        asset_symbol = f"{trade_asset}USDT"
        df_asset = fetch_binance_klines(symbol=asset_symbol)
        
        if not df_asset.empty:
            df_asset = df_asset[['open_time', 'close']].rename(columns={'close': 'asset_price'})
            df_test = pd.merge(df_test, df_asset, on='open_time', how='left')
            df_test['asset_price'] = df_test['asset_price'].ffill()
        else:
            # 修改回傳值數量，保持一致 (多加一個 0)
            return pd.DataFrame(), pd.DataFrame(), 0, 0, 0
            
    df_test = df_test[df_test['open_time'].dt.date >= start_date]
    if df_test.empty: return pd.DataFrame(), pd.DataFrame(), 0, 0, 0 # 修改回傳值數量

    asset_balance = 0      
    total_invested = 0     
    realized_pnl = 0
    total_fees = 0 # 累計手續費
    
    trade_history = []
    portfolio_history = []
    buy_days = 0
    sell_days = 0

    for index, row in df_test.iterrows():
        price = row[price_col]
        risk = row['total_risk']
        date = row['open_time']
        
        if pd.isna(price) or price <= 0: continue
            
        action = None
        trade_val = 0
        trade_amount = 0
        fee_amount = 0
        
        # Buy
        if buy_min <= risk < buy_max:
            buy_days += 1
            action = "BUY"
            
            # 手續費計算 (Binance Spot: 扣除手續費後的淨投資額)
            fee_amount = buy_amount * fee_rate
            net_invest = buy_amount - fee_amount
            trade_amount = net_invest / price
            
            asset_balance += trade_amount
            total_invested += buy_amount # 總投入本金還是 buy_amount
            total_fees += fee_amount
            
            trade_val = buy_amount
            
        # Sell
        elif sell_min <= risk < sell_max:
            sell_days += 1
            if asset_balance > 0:
                action = "SELL"
                amount_to_sell = asset_balance * sell_pct
                if amount_to_sell > 0:
                    gross_val = amount_to_sell * price
                    fee_amount = gross_val * fee_rate
                    net_val = gross_val - fee_amount # 實際拿到手的 USDT
                    
                    avg_cost = total_invested / asset_balance if asset_balance > 0 else 0
                    cost_of_sold = amount_to_sell * avg_cost
                    
                    asset_balance -= amount_to_sell
                    total_invested -= cost_of_sold 
                    
                    # 損益 = 淨回收額 - 成本
                    realized_pnl += (net_val - cost_of_sold) 
                    total_fees += fee_amount
                    
                    trade_val = net_val
                    trade_amount = amount_to_sell

        market_value = asset_balance * price
        unrealized_pnl = market_value - total_invested
        total_equity = market_value + realized_pnl 
        current_avg_cost = total_invested / asset_balance if asset_balance > 0 else 0
        
        if action:
            trade_history.append({
                'Date': date, 'Type': action, 'Price': price, 'Risk': risk,
                'Val_USDT': trade_val, 'Amount': trade_amount, 'Fee': fee_amount, 'Balance': asset_balance
            })
            
        # 計算當前最大權益 (for MDD)
        peak_equity = max(portfolio_history[-1]['Equity'] if portfolio_history else total_equity, total_equity)
        
        portfolio_history.append({
            'Date': date, 
            'Equity': total_equity, 
            'Invested': total_invested,
            'Realized_PnL': realized_pnl, 
            'Unrealized_PnL': unrealized_pnl,
            'Total_Fees': total_fees,
            'Avg_Cost': current_avg_cost,
            'Peak_Equity': peak_equity
        })
    final_price = df_test.iloc[-1]['asset_price'] if not df_test.empty else 0

    return pd.DataFrame(trade_history), pd.DataFrame(portfolio_history), buy_days, sell_days, final_price

def run_portfolio_backtest(df_risk, asset_weights, total_daily_buy, buy_min, buy_max, sell_pct, sell_min, sell_max, start_date, fee_rate=0.001):
    all_prices = {}
    for asset in asset_weights.keys():
        if asset == "BTC": all_prices[asset] = df_risk.set_index('open_time')['close']
        else:
            df_asset = fetch_binance_klines(symbol=f"{asset}USDT")
            if not df_asset.empty: all_prices[asset] = df_asset.set_index('open_time')['close']

    df_test = df_risk[df_risk['open_time'].dt.date >= start_date].copy()
    if df_test.empty: return pd.DataFrame(), {}, {}, 0, 0 

    asset_results = {asset: {
        'balance': 0.0, 'cum_invested': 0.0, 'current_cost': 0.0, 
        'realized_pnl': 0.0, 'fees': 0.0,
        'peak': 0.0, 'mdd': 0.0 
    } for asset in asset_weights.keys()}
    
    portfolio_history = []
    
    # --- 新增計數器 ---
    total_buy_days = 0
    total_sell_days = 0
    
    for _, row in df_test.iterrows():
        date = row['open_time']
        risk = row['total_risk']
        is_buy, is_sell = buy_min <= risk < buy_max, sell_min <= risk < sell_max
        daily_mkt_val = 0.0

        # --- 統計天數 ---
        if is_buy: total_buy_days += 1
        if is_sell: total_sell_days += 1

        for asset, weight in asset_weights.items():
            if asset not in all_prices or date not in all_prices[asset].index: continue
            price = all_prices[asset].loc[date]
            res = asset_results[asset]
            
            if is_buy:
                budget = total_daily_buy * weight
                fee = budget * fee_rate
                res['balance'] += (budget - fee) / price
                res['cum_invested'] += budget
                res['current_cost'] += budget
                res['fees'] += fee
            elif is_sell and res['balance'] > 0:
                amt_to_sell = res['balance'] * sell_pct
                val = amt_to_sell * price
                fee, cost_ratio = val * fee_rate, amt_to_sell / res['balance']
                sold_cost = res['current_cost'] * cost_ratio
                res['realized_pnl'] += (val - fee - sold_cost)
                res['balance'] -= amt_to_sell
                res['current_cost'] -= sold_cost
                res['fees'] += fee

            asset_equity = (res['balance'] * price) + res['realized_pnl']
            if asset_equity > res['peak']: res['peak'] = asset_equity
            if res['peak'] > 0:
                dd = (asset_equity - res['peak']) / res['peak']
                if dd < res['mdd']: res['mdd'] = dd
            
            daily_mkt_val += (res['balance'] * price)

        total_realized = sum(r['realized_pnl'] for r in asset_results.values())
        portfolio_history.append({
            'Date': date, 'Equity': daily_mkt_val + total_realized,
            'Total_Cost': sum(r['cum_invested'] for r in asset_results.values()),
            'Realized': total_realized
        })
        
    return pd.DataFrame(portfolio_history), asset_results, all_prices, total_buy_days, total_sell_days

import json

def generate_interactive_html(df):
    chart_data = []
    
    min_price = df['close'].min()
    log_min = np.log10(min_price)
    
    # 成交量標準化
    vol_min = df['volume'].min()
    vol_max = df['volume'].max()
    
    gradient_stops = []
    total_points = len(df)
    
    halving_dates = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]
    halving_indices = []
    
    for i, row in enumerate(df.itertuples()):
        date_str = row.open_time.strftime("%Y-%m-%d")
        if date_str in halving_dates:
            halving_indices.append(i)
            
        log_price = np.log10(row.close)
        y_pos = (log_price - log_min) * 120 
        z_pos = (0.5 - row.total_risk) * 60 
        
        # --- 成交量柱子計算 ---
        raw_vol_ratio = (row.volume - vol_min) / (vol_max - vol_min)
        vol_power = raw_vol_ratio ** 3 
        
        # 【修改 1】半徑稍微調小一點 (原本 15.0 -> 10.0)，適應較短的時間軸
        radius = 0.1 + (vol_power * 10.0)
        
        opacity = 0.1 + (vol_power * 0.9)
        height_scale = 0.8 + (vol_power * 0.2)
        
        if row.total_risk > 0.6:
            color_hex = "#ff9100" 
            risk_color_css = "rgba(255, 145, 0, 1)"
        elif row.total_risk < 0.4:
            color_hex = "#00e5ff" 
            risk_color_css = "rgba(0, 229, 255, 1)"
        else:
            color_hex = "#666666" 
            risk_color_css = "rgba(100, 100, 100, 0)"

        is_signal = False
        signal_type = "none"
        if row.total_risk < 0.4:
            is_signal = True; signal_type = "buy"
        elif row.total_risk >= 0.8:
            is_signal = True; signal_type = "sell"

        chart_data.append({
            # 【修改 2】X軸間距縮短 (原本 4.0 -> 1.5)
            # 數值越小，整體時間軸越短；數值越大，拉得越長
            "x": i * 1.5,       
            "y": y_pos,
            "z": z_pos,
            "radius": radius,   
            "opacity": opacity, 
            "height_scale": height_scale,
            "date": date_str,
            "price": f"${row.close:,.0f}",
            "risk": f"{row.total_risk:.2f}",
            "volume": f"{row.volume:,.0f}",
            "color": color_hex,
            "is_signal": is_signal,
            "signal_type": signal_type
        })
        
        if i % 5 == 0 or i == total_points - 1:
            pct = round((i / total_points) * 100, 2)
            gradient_stops.append(f"{risk_color_css} {pct}%")

    json_data = json.dumps(chart_data)
    halving_data = json.dumps(halving_indices)
    gradient_css = ",".join(gradient_stops)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>BTC Volume Pillars 3D</title>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #000; font-family: 'Segoe UI', monospace; }}
            
            #ui-layer {{
                position: absolute; top: 20px; left: 20px; color: #fff;
                background: rgba(0, 0, 0, 0.85); padding: 20px; 
                border-left: 4px solid #00e5ff; border-radius: 8px;
                pointer-events: none; user-select: none; z-index: 10;
                box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
            }}
            h1 {{ margin: 0 0 10px 0; font-size: 22px; color: #00e5ff; letter-spacing: 1px; }}
            .key {{ display: flex; align-items: center; margin-bottom: 5px; font-size: 13px; color: #ddd; }}
            .dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }}
            .dot-buy {{ background-color: #00e5ff; box-shadow: 0 0 8px #00e5ff; }}
            .dot-sell {{ background-color: #ff9100; box-shadow: 0 0 8px #ff9100; }}
            .pillar {{ width: 6px; height: 16px; background: linear-gradient(to top, transparent, #fff); margin-right: 10px; }}
            .line-wall {{ width: 20px; height: 10px; border: 1px solid rgba(255,255,255,0.5); background: rgba(255,255,255,0.1); margin-right: 5px; }}
            .controls {{ color: #888; font-size: 11px; margin-top: 15px; line-height: 1.5; border-top: 1px solid #333; padding-top: 10px; }}
            .highlight {{ color: #fff; font-weight: bold; }}

            #hud-panel {{
                position: absolute; top: 20px; right: 20px; width: 220px;
                background: rgba(0, 15, 30, 0.9); 
                border: 1px solid #444; border-radius: 5px;
                padding: 15px; color: #00e5ff; z-index: 10;
                pointer-events: none;
            }}
            #hud-title {{ font-size: 12px; border-bottom: 1px solid #333; margin-bottom: 8px; padding-bottom: 5px; color: #aaa; letter-spacing: 1px; }}
            .hud-val {{ font-size: 18px; font-weight: bold; color: #fff; text-shadow: 0 0 5px rgba(255,255,255,0.5); display: block; margin-bottom: 8px; }}
            .hud-sub {{ font-size: 12px; color: #888; }}

            #crosshair {{
                position: absolute; top: 50%; left: 50%; width: 20px; height: 20px;
                transform: translate(-50%, -50%); pointer-events: none; z-index: 20; opacity: 0.8;
            }}
            #crosshair::before {{ content: ''; position: absolute; background: #00e5ff; top: 9px; left: 0; width: 20px; height: 2px; box-shadow: 0 0 4px #00e5ff; }}
            #crosshair::after {{ content: ''; position: absolute; background: #00e5ff; top: 0; left: 9px; width: 2px; height: 20px; box-shadow: 0 0 4px #00e5ff; }}

            #timeline-container {{
                position: absolute; bottom: 30px; left: 5%; width: 90%;
                height: 60px; z-index: 100;
                display: flex; flex-direction: column; align-items: center;
                background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
                pointer-events: auto;
            }}
            #date-display {{ font-size: 16px; font-weight: bold; color: #fff; margin-bottom: 5px; text-shadow: 0 0 5px #000; }}
            
            #timeline-slider {{
                -webkit-appearance: none; width: 100%; height: 8px;
                border-radius: 4px; outline: none;
                background: linear-gradient(to right, {gradient_css});
                box-shadow: 0 0 10px rgba(0,0,0,0.5); cursor: pointer;
            }}
            #timeline-slider::-webkit-slider-thumb {{
                -webkit-appearance: none; appearance: none;
                width: 16px; height: 16px; border-radius: 50%;
                background: #fff; border: 2px solid #000; cursor: pointer;
                transition: transform 0.1s;
            }}
            #timeline-slider::-webkit-slider-thumb:hover {{ transform: scale(1.3); }}
            
            #lock-msg {{
                position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                color: #00e5ff; background: rgba(0,0,0,0.9); padding: 30px 50px;
                border: 2px solid #00e5ff; border-radius: 4px;
                text-align: center; display: none; pointer-events: none; z-index: 50;
                box-shadow: 0 0 30px rgba(0, 229, 255, 0.2);
            }}
        </style>
    </head>
    <body>
        <div id="crosshair"></div>
        
        <div id="ui-layer">
            <h1>BTC VOLUME PILLARS</h1>
            <div class="key"><div class="line-wall"></div> 巨大光牆 = 減半週期</div>
            <div class="key"><div class="pillar"></div> 柱子粗細 = 成交量爆發</div>
            <div class="key"><div class="dot dot-buy"></div> 右側 (青) = 低風險區</div>
            <div class="key"><div class="dot dot-sell"></div> 左側 (橘) = 高風險區</div>
            <div class="controls">
                <span class="highlight">點擊中央</span> 鎖定滑鼠<br>
                <span class="highlight">WASD</span> 移動 | <span class="highlight">SPACE</span> 上升
            </div>
        </div>

        <div id="hud-panel">
            <div id="hud-title">TARGET DATA</div>
            <span class="hud-sub">DATE</span>
            <span class="hud-val" id="hud-date">--</span>
            <span class="hud-sub">PRICE / VOL</span>
            <span class="hud-val" id="hud-price">--</span>
            <span class="hud-sub">RISK LEVEL</span>
            <span class="hud-val" id="hud-risk">--</span>
        </div>

        <div id="timeline-container">
            <div id="date-display">Drag to Time Travel</div>
            <input type="range" id="timeline-slider" min="0" max="{len(chart_data)-1}" value="0">
        </div>

        <div id="lock-msg">
            <h2 style="margin:0 0 10px 0;">SYSTEM PAUSED</h2>
            <p style="margin:0;">CLICK TO RESUME FLIGHT</p>
        </div>

        <script type="module">
            import * as THREE from 'https://unpkg.com/three@0.126.0/build/three.module.js';
            import {{ PointerLockControls }} from 'https://unpkg.com/three@0.126.0/examples/jsm/controls/PointerLockControls.js';

            let camera, scene, renderer, controls, raycaster;
            let moveForward = false, moveBackward = false, moveLeft = false, moveRight = false, moveUp = false, moveDown = false;
            let prevTime = performance.now();
            let velocity = new THREE.Vector3();
            let direction = new THREE.Vector3();
            let isDraggingSlider = false;
            
            const dataPoints = {json_data};
            const halvingIndices = {halving_data}; 
            
            const slider = document.getElementById('timeline-slider');
            const dateDisplay = document.getElementById('date-display');
            const lockMsg = document.getElementById('lock-msg');
            const hudDate = document.getElementById('hud-date');
            const hudPrice = document.getElementById('hud-price');
            const hudRisk = document.getElementById('hud-risk');
            const hudPanel = document.getElementById('hud-panel');
            const uiLayer = document.getElementById('ui-layer');

            const interactableObjects = [];

            init();
            animate();

            function init() {{
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x050508); 
                scene.fog = new THREE.FogExp2(0x050508, 0.0012);

                camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 5000);
                updateCameraPosition(0);

                const gridHelper = new THREE.GridHelper(8000, 200, 0x111111, 0x080808);
                scene.add(gridHelper);

                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
                dirLight.position.set(0, 200, 100);
                scene.add(dirLight);

                // --- 1. 主軌跡線 ---
                const positions = []; const colors = []; const colorObj = new THREE.Color();
                dataPoints.forEach(pt => {{
                    positions.push(pt.x, pt.y, pt.z);
                    colorObj.set(pt.color);
                    colors.push(colorObj.r, colorObj.g, colorObj.b);
                }});
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                const material = new THREE.LineBasicMaterial({{ vertexColors: true, linewidth: 2, opacity: 0.9, transparent: true }});
                scene.add(new THREE.Line(geometry, material));

                // 隱形感應區
                const hitGeo = new THREE.BoxGeometry(4, 100, 4); 
                const hitMat = new THREE.MeshBasicMaterial({{ visible: false }});

                dataPoints.forEach((pt, idx) => {{
                    const mesh = new THREE.Mesh(hitGeo, hitMat);
                    mesh.position.set(pt.x, pt.y, pt.z);
                    mesh.userData = {{ 
                        date: pt.date, 
                        price: `${{pt.price}} / Vol: ${{pt.volume}}`, 
                        risk: pt.risk, 
                        color: pt.color 
                    }};
                    scene.add(mesh);
                    interactableObjects.push(mesh);

                    // --- 2. 成交量爆發柱 (Volume Burst Pillars) ---
                    // 間隔 7
                    if (idx % 7 === 0) {{
                        const floorY = -50; 
                        const targetHeight = (pt.y - floorY) * pt.height_scale;
                        
                        const pillarGeo = new THREE.CylinderGeometry(pt.radius, pt.radius, targetHeight, 8);
                        const pillarMat = new THREE.MeshBasicMaterial({{ 
                            color: pt.color, 
                            transparent: true, 
                            opacity: pt.opacity,
                            wireframe: false
                        }});
                        
                        const pillar = new THREE.Mesh(pillarGeo, pillarMat);
                        pillar.position.set(pt.x, floorY + targetHeight/2, pt.z);
                        scene.add(pillar);
                    }}

                    // --- 3. 訊號球 ---
                    if (pt.is_signal) {{
                        if (idx % 3 === 0) {{
                            const sphereGeo = new THREE.IcosahedronGeometry(pt.signal_type === 'buy' ? 3.5 : 3.0, 0); 
                            const mat = new THREE.MeshBasicMaterial({{ color: pt.color, wireframe: true }});
                            const mesh = new THREE.Mesh(sphereGeo, mat);
                            mesh.position.set(pt.x, pt.y, pt.z);
                            scene.add(mesh);
                        }}
                    }}
                }});

                // --- 4. 減半週期光牆 ---
                const wallGeo = new THREE.PlaneGeometry(300, 2000); 
                const wallMat = new THREE.MeshBasicMaterial({{ 
                    color: 0xffffff, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false
                }});

                halvingIndices.forEach(idx => {{
                    if (idx < dataPoints.length) {{
                        const pt = dataPoints[idx];
                        const wall = new THREE.Mesh(wallGeo, wallMat);
                        wall.position.set(pt.x, 500, 0); 
                        wall.rotation.y = Math.PI / 2;
                        scene.add(wall);
                        
                        const labelGeo = new THREE.CylinderGeometry(1, 1, 200, 32);
                        const labelMat = new THREE.MeshBasicMaterial({{ color: 0xffffff, opacity: 0.5, transparent: true }});
                        const label = new THREE.Mesh(labelGeo, labelMat);
                        label.position.set(pt.x, 0, -40);
                        scene.add(label);
                    }}
                }});

                raycaster = new THREE.Raycaster();
                raycaster.far = 1000;

                controls = new PointerLockControls(camera, document.body);
                
                document.body.addEventListener('click', (e) => {{
                    if (e.target !== slider && !isDraggingSlider) controls.lock();
                }});

                controls.addEventListener('lock', () => {{
                    lockMsg.style.display = 'none';
                    hudPanel.style.opacity = '1';
                    uiLayer.style.opacity = '0.3';
                }});
                
                controls.addEventListener('unlock', () => {{
                    if (!isDraggingSlider) lockMsg.style.display = 'block';
                    hudPanel.style.opacity = '0.5';
                    uiLayer.style.opacity = '1';
                }});

                slider.addEventListener('mousedown', () => {{ isDraggingSlider = true; lockMsg.style.display = 'none'; }});
                slider.addEventListener('mouseup', () => {{ isDraggingSlider = false; lockMsg.style.display = 'block'; }});

                slider.addEventListener('input', (e) => {{
                    const idx = parseInt(e.target.value);
                    const pt = dataPoints[idx];
                    dateDisplay.innerText = `${{pt.date}} | Price: ${{pt.price}} | Risk: ${{pt.risk}}`;
                    dateDisplay.style.color = pt.color;
                    updateCameraPosition(idx);
                }});

                const onKeyDown = (event) => {{
                    switch (event.code) {{
                        case 'ArrowUp': case 'KeyW': moveForward = true; break;
                        case 'ArrowLeft': case 'KeyA': moveLeft = true; break;
                        case 'ArrowDown': case 'KeyS': moveBackward = true; break;
                        case 'ArrowRight': case 'KeyD': moveRight = true; break;
                        case 'Space': moveUp = true; break;
                        case 'ShiftLeft': moveDown = true; break;
                    }}
                }};
                const onKeyUp = (event) => {{
                    switch (event.code) {{
                        case 'ArrowUp': case 'KeyW': moveForward = false; break;
                        case 'ArrowLeft': case 'KeyA': moveLeft = false; break;
                        case 'ArrowDown': case 'KeyS': moveBackward = false; break;
                        case 'ArrowRight': case 'KeyD': moveRight = false; break;
                        case 'Space': moveUp = false; break;
                        case 'ShiftLeft': moveDown = false; break;
                    }}
                }};
                document.addEventListener('keydown', onKeyDown);
                document.addEventListener('keyup', onKeyUp);

                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setPixelRatio(window.devicePixelRatio);
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);
                window.addEventListener('resize', onWindowResize);
            }}

            function updateCameraPosition(index) {{
                const pt = dataPoints[index];
                camera.position.set(pt.x - 50, pt.y + 30, 0); 
                camera.lookAt(pt.x + 100, pt.y, 0); 
            }}

            function onWindowResize() {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }}

            function updateHUD() {{
                raycaster.setFromCamera(new THREE.Vector2(0, 0), camera);
                const intersects = raycaster.intersectObjects(interactableObjects);
                if (intersects.length > 0) {{
                    const hit = intersects[0].object;
                    const data = hit.userData;
                    hudDate.innerText = data.date;
                    hudPrice.innerText = data.price;
                    hudRisk.innerText = data.risk;
                    hudRisk.style.color = data.color;
                    hudPanel.style.borderColor = data.color;
                }}
            }}

            function animate() {{
                requestAnimationFrame(animate);
                const time = performance.now();
                if (controls.isLocked) {{
                    const delta = (time - prevTime) / 1000;
                    velocity.x -= velocity.x * 10.0 * delta;
                    velocity.z -= velocity.z * 10.0 * delta;
                    velocity.y -= velocity.y * 10.0 * delta;
                    direction.z = Number(moveForward) - Number(moveBackward);
                    direction.x = Number(moveRight) - Number(moveLeft);
                    direction.y = Number(moveDown) - Number(moveUp);
                    direction.normalize();

                    if (moveForward || moveBackward) velocity.z -= direction.z * 800.0 * delta;
                    if (moveLeft || moveRight) velocity.x -= direction.x * 800.0 * delta;
                    if (moveUp || moveDown) velocity.y -= direction.y * 800.0 * delta;

                    controls.moveRight(-velocity.x * delta);
                    controls.moveForward(-velocity.z * delta);
                    camera.position.y += velocity.y * delta;
                    updateHUD();
                }}
                prevTime = time;
                renderer.render(scene, camera);
            }}
        </script>
    </body>
    </html>
    """
    return html_content
# ============================================================
# 6. HTML 生成函數 (疊加模式 - ATH 對齊)
# ============================================================
def generate_overlay_html(df):
    # 1. 數據檢查
    if df.empty: return "<h3>NO DATA</h3>"
    df = df.fillna(0)
    
    # 2. 定義週期 (ATH 熊市對齊)
    cycles = [
        {"start": "2013-11-30", "end": "2017-12-17", "name": "2014 CYCLE", "z": -100, "color": "#555555", "is_current": False},
        {"start": "2017-12-17", "end": "2021-11-10", "name": "2018 CYCLE", "z": -50,  "color": "#777777", "is_current": False},
        {"start": "2021-11-10", "end": "2099-12-31", "name": "CURRENT",    "z": 0,    "color": "#ffffff", "is_current": True},
    ]
    
    chart_data = []
    vol_series = df['volume'].replace(0, 1)
    vol_min = np.log10(vol_series).min()
    vol_max = np.log10(vol_series).max()

    for cycle in cycles:
        mask = (df['open_time'] >= cycle['start']) & (df['open_time'] < cycle['end'])
        cycle_df = df.loc[mask].copy().reset_index(drop=True)
        if cycle_df.empty: continue
        
        # 基準價格 = ATH
        base_price = cycle_df.iloc[0]['close']
        if base_price <= 0: base_price = 1
        
        for i, row in enumerate(cycle_df.itertuples()):
            # X軸: 距離 ATH 的天數
            x_pos = i * 1.5
            
            # Y軸: 回撤幅度 (Drawdown)
            drawdown = (row.close - base_price) / base_price
            y_pos = drawdown * 150 
            
            # Z軸: 分層 + 風險
            risk_val = float(getattr(row, 'total_risk', 0.5))
            z_pos = cycle['z'] + (0.5 - risk_val) * 20
            
            # 成交量
            log_vol = np.log10(row.volume if row.volume > 0 else 1)
            vol_ratio = (log_vol - vol_min) / (vol_max - vol_min) if (vol_max - vol_min) > 0 else 0
            vol_power = vol_ratio ** 3
            
            radius = (0.3 if cycle['is_current'] else 0.1) + (vol_power * 6.0)
            opacity = 0.3 + (vol_power * 0.7)
            height_scale = 0.5 + (vol_power * 0.5)
            
            # 顏色
            if cycle['is_current']:
                if risk_val >= 0.8: color_hex = "#ff9100" 
                elif risk_val <= 0.4: color_hex = "#00e5ff" 
                else: color_hex = "#ffffff"
            else:
                color_hex = cycle['color'] # 歷史週期用灰階

            is_signal = False
            signal_type = "none"
            if cycle['is_current']:
                if risk_val <= 0.4: is_signal = True; signal_type = "buy"
                elif risk_val >= 0.8: is_signal = True; signal_type = "sell"

            chart_data.append({
                "x": float(x_pos), "y": float(y_pos), "z": float(z_pos),
                "radius": float(radius), "opacity": float(opacity),
                "date": row.open_time.strftime("%Y-%m-%d"),
                "price": f"${row.close:,.0f}", "risk": f"{risk_val:.2f}",
                "roi": f"{drawdown*100:.2f}%", "cycle": cycle['name'],
                "color": color_hex, "is_signal": is_signal, "signal_type": signal_type,
                "is_current": cycle['is_current']
            })

    json_data = json.dumps(chart_data)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>BTC 3D Overlay</title>
        <style>
            body {{ margin: 0; overflow: hidden; background-color: #000010; font-family: monospace; }}
            #ui-layer {{ position: absolute; top: 20px; left: 20px; color: #fff; background: rgba(0,0,0,0.8); padding: 15px; border: 1px solid #fff; pointer-events: none; }}
            #hud-panel {{ position: absolute; top: 20px; right: 20px; color: #00ff00; background: rgba(0,0,0,0.8); padding: 15px; border: 1px solid #00ff00; font-size: 14px; pointer-events: none; }}
            #lock-msg {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; background: rgba(0,0,0,0.8); padding: 20px; border: 1px solid white; text-align: center; display: block; pointer-events: none; z-index: 50; }}
            #crosshair {{ position: absolute; top: 50%; left: 50%; width: 10px; height: 10px; background: red; transform: translate(-50%, -50%); border-radius: 50%; pointer-events: none; z-index: 20; }}
        </style>
        <script type="importmap">
        {{ "imports": {{ "three": "https://unpkg.com/three@0.160.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" }} }}
        </script>
    </head>
    <body>
        <div id="crosshair"></div>
        <div id="ui-layer"><h2>ATH DRAWDOWN MODE</h2><p>WASD = Move | SPACE = Up | SHIFT = Down</p><p>Click Center to Start</p></div>
        <div id="hud-panel"><div>CYCLE: <span id="h-cycle">--</span></div><div>DATE: <span id="h-date">--</span></div><div>ROI: <span id="h-price">--</span></div></div>
        <div id="lock-msg"><h2>CLICK TO START</h2></div>
        <script type="module">
            import * as THREE from 'three';
            import {{ PointerLockControls }} from 'three/addons/controls/PointerLockControls.js';

            let camera, scene, renderer, controls, raycaster;
            let moveForward=false, moveBackward=false, moveLeft=false, moveRight=false;
            let moveUp=false, moveDown=false;
            let prevTime = performance.now();
            let velocity = new THREE.Vector3();
            let direction = new THREE.Vector3();

            const dataPoints = {json_data};
            const interactableObjects = [];
            const hCycle = document.getElementById('h-cycle');
            const hDate = document.getElementById('h-date');
            const hPrice = document.getElementById('h-price');
            const lockMsg = document.getElementById('lock-msg');

            init();
            animate();

            function init() {{
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x050505); 
                scene.fog = new THREE.Fog(0x050505, 10, 3000);
                camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 1, 8000);
                camera.position.set(-150, 100, 300);
                camera.lookAt(200, 0, -100);

                const grid = new THREE.GridHelper(5000, 100, 0x333333, 0x111111);
                grid.position.y = -100; scene.add(grid);

                const athLineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-2000, 0, -300), new THREE.Vector3(5000, 0, 100)]);
                const athLineMat = new THREE.LineBasicMaterial({{ color: 0xffffff, opacity: 0.3, transparent: true }});
                scene.add(new THREE.Line(athLineGeo, athLineMat));

                scene.add(new THREE.AmbientLight(0xffffff, 0.8));
                const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
                dirLight.position.set(0, 500, 200);
                scene.add(dirLight);

                const geo = new THREE.CylinderGeometry(1, 1, 1, 6);
                const hitGeo = new THREE.BoxGeometry(2, 200, 2);
                const hitMat = new THREE.MeshBasicMaterial({{ visible: false }});

                dataPoints.forEach((pt, idx) => {{
                    const hit = new THREE.Mesh(hitGeo, hitMat);
                    hit.position.set(pt.x, pt.y, pt.z);
                    hit.userData = pt;
                    scene.add(hit);
                    interactableObjects.push(hit);

                    if (idx % 3 === 0) {{
                        const floorY = -100;
                        const h = Math.max(1, pt.y - floorY);
                        const mat = new THREE.MeshBasicMaterial({{ color: pt.color, transparent: true, opacity: pt.opacity }});
                        const mesh = new THREE.Mesh(geo, mat);
                        mesh.position.set(pt.x, floorY + h/2, pt.z);
                        mesh.scale.set(pt.radius, h, pt.radius);
                        scene.add(mesh);
                    }}
                    
                    if (pt.is_current && pt.is_signal && idx % 3 === 0) {{
                        const sMat = new THREE.MeshBasicMaterial({{ color: pt.color, wireframe: true }});
                        const sGeo = new THREE.IcosahedronGeometry(3, 0);
                        const sphere = new THREE.Mesh(sGeo, sMat);
                        sphere.position.set(pt.x, pt.y, pt.z);
                        scene.add(sphere);
                    }}
                }});

                raycaster = new THREE.Raycaster();
                controls = new PointerLockControls(camera, document.body);
                document.body.addEventListener('click', () => controls.lock());
                controls.addEventListener('lock', () => lockMsg.style.display = 'none');
                controls.addEventListener('unlock', () => lockMsg.style.display = 'block');

                const onKeyDown = (e) => {{
                    switch (e.code) {{ 
                        case 'KeyW': moveForward=true; break; 
                        case 'KeyA': moveLeft=true; break; 
                        case 'KeyS': moveBackward=true; break; 
                        case 'KeyD': moveRight=true; break; 
                        case 'Space': moveUp=true; break; 
                        case 'ShiftLeft': moveDown=true; break; 
                    }}
                }};
                const onKeyUp = (e) => {{
                    switch (e.code) {{ 
                        case 'KeyW': moveForward=false; break; 
                        case 'KeyA': moveLeft=false; break; 
                        case 'KeyS': moveBackward=false; break; 
                        case 'KeyD': moveRight=false; break; 
                        case 'Space': moveUp=false; break; 
                        case 'ShiftLeft': moveDown=false; break; 
                    }}
                }};
                document.addEventListener('keydown', onKeyDown);
                document.addEventListener('keyup', onKeyUp);

                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);
                window.addEventListener('resize', () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            }}

            function animate() {{
                requestAnimationFrame(animate);
                if (controls.isLocked) {{
                    const time = performance.now();
                    const delta = (time - prevTime) / 1000;
                    velocity.x -= velocity.x * 10.0 * delta;
                    velocity.z -= velocity.z * 10.0 * delta;
                    velocity.y -= velocity.y * 10.0 * delta;
                    direction.z = Number(moveForward) - Number(moveBackward);
                    direction.x = Number(moveRight) - Number(moveLeft);
                    direction.y = Number(moveDown) - Number(moveUp);
                    direction.normalize();
                    if (moveForward || moveBackward) velocity.z -= direction.z * 1000.0 * delta;
                    if (moveLeft || moveRight) velocity.x -= direction.x * 1000.0 * delta;
                    if (moveUp || moveDown) velocity.y -= direction.y * 1000.0 * delta;
                    controls.moveRight(-velocity.x * delta);
                    controls.moveForward(-velocity.z * delta);
                    camera.position.y += velocity.y * delta;

                    raycaster.setFromCamera(new THREE.Vector2(0,0), camera);
                    const intersects = raycaster.intersectObjects(interactableObjects);
                    if(intersects.length > 0) {{
                        const d = intersects[0].object.userData;
                        hCycle.innerText = d.cycle;
                        hDate.innerText = d.date;
                        hPrice.innerText = d.roi;
                        hPrice.style.color = d.color;
                    }}
                    prevTime = time;
                }} else {{
                    prevTime = performance.now();
                }}
                renderer.render(scene, camera);
            }}
        </script>
    </body>
    </html>
    """
    return html_content
# ============================================================
# 6. 主介面 (Main) - Tab 2 改為分頁版
# ============================================================

def main():
    # 1. 🚀 專業戰情室 CSS 整合方案 (亮度與對比極大化)
    st.markdown("""
        <style>
        /* (1) 左側側邊欄：徹底純白增亮修正 */
        [data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #2d3139; }
        
        /* 強制側邊欄所有文字、Markdown、標籤、按鈕字體變純白 */
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] label p, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] summary {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            font-size: 1.05rem !important;
        }

        /* 修正同步按鈕：深色背景 + 亮青邊框 (解決反白問題) */
        [data-testid="stSidebar"] .stButton button {
            background-color: #1c212d !important;
            color: #00FFC2 !important; 
            border: 1px solid #00FFC2 !important;
            border-radius: 10px !important;
            width: 100%;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: #00FFC2 !important;
            color: #000000 !important;
            box-shadow: 0 0 15px rgba(0, 255, 194, 0.4);
        }

        /* (2) 🎯 核心深色卡片：標籤純白 + 數字電磁亮青發光 */
        .metric-card {
            background-color: #05070a;
            border: 1px solid #4a5162; 
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8);
            margin-bottom: 15px;
            
            height: 180px;            /* 強制三張卡片高度完全一致 */
            display: flex;
            flex-direction: column;
            justify-content: center;  /* 內容垂直置中 */
            align-items: center;      /* 內容水平置中 */
        }
        /* 中英文標籤變純白色 */
        .metric-card .metric-label, .metric-card .metric-sub { 
            color: #FFFFFF !important; 
            font-weight: 700 !important;
            opacity: 1 !important;
        }
        /* 數字改為「電磁亮青」並加強發光 */
        .metric-card .metric-value { 
            font-size: 38px; 
            font-weight: 900;
            color: #00FFC2 !important; 
            font-family: 'JetBrains Mono', monospace;
            text-shadow: 0 0 15px rgba(0, 255, 194, 0.7); 
            margin: 10px 0px;
        }
       /* 進度條容器 */
        .risk-progress-container {
            background-color: #1c212d !important;
            border-radius: 10px;
            height: 12px !important;
            width: 100% !important;
            margin-top: 15px;
            border: 1px solid #2d3139;
            overflow: hidden;
            position: relative; /* 確保內部定位正確 */
        }

        /* 進度條彩色部分 */
        .risk-progress-fill {
            height: 100% !important;   /* 高度設為 100% 填滿容器 */
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 255, 194, 0.6);
            transition: width 0.8s ease-in-out;
            display: block !important;
        }
        /* (3) 亮色區域強力穿透 (強制白底黑字) */
        div:has(> .force-light) + div [data-testid="stExpander"],
        div:has(> .force-light) + div [data-testid="stExpander"] details,
        div:has(> .force-light) + div [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
            background-color: #FFFFFF !important;
            border-color: #DCDFE6 !important;
        }
        div:has(> .force-light) + div [data-testid="stExpander"] * {
            color: #111827 !important;
            text-shadow: none !important;
        }
        .force-light { display: none; }
        </style>
    """, unsafe_allow_html=True)

    # 2. 數據加載 (保留原邏輯)
    with st.spinner('🎯 正在同步全球數據...'):
        df = load_data_and_compute()
    if df.empty: return

   # --- 修正區：補齊所有變數定義 ---
    latest = df.iloc[-1]
    curr_risk = latest['total_risk']
    curr_price = latest['close']
    
    # 取得「昨天的數據」用來計算 Delta
    if len(df) > 1:
        prev_row = df.iloc[-2]
        prev_risk = prev_row['total_risk']
        prev_close = prev_row['close']
    else:
        prev_risk = curr_risk
        prev_close = curr_price

    # 計算價格漲跌幅
    price_pct = ((curr_price - prev_close) / prev_close) * 100
    
    # 【關鍵：補齊 status_msg 邏輯】
    if curr_risk <= 0.4:
        advice_title = " 💎 BUY"
        advice_color = "#00e676"
        status_msg = "抄底窗口" # 補上這一行
    elif curr_risk >= 0.8:
        advice_title = " 🔥 SELL"
        advice_color = "#ff5252"
        status_msg = "風險警戒" # 補上這一行
    else:
        advice_title = " 🧘 HODL"
        advice_color = "#ffeb3b"
        status_msg = "趨勢運行中" # 補上這一行

    # 狀態燈顏色與策略建議顏色連動
    status_color = advice_color

    with st.sidebar:
        # --- 頂部品牌區 ---
        st.markdown(f"""
            <div style="padding: 10px 0px;">
                <h2 style='color: white; margin-bottom:0;'>BTC Cycle</h2>
                <p style='color: gray; font-size: 12px;'>Professional Quantitative Tool</p>
            </div>
            <div style="background: #1d212d; padding: 15px; border-radius: 10px; border: 1px solid {status_color}33; margin-bottom: 20px;">
                <span class="status-light" style="background-color: {status_color}; box-shadow: 0 0 10px {status_color};"></span>
                <span style="color: {status_color}; font-weight: bold; font-size: 14px;">{status_msg}</span>
                <br><span style="color: gray; font-size: 11px;">Current Risk: {curr_risk:.2f}</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ 分析模組")
        # 這裡保留你原本的內容標籤，但加上 Emoji 讓視覺更直覺
        menu = st.radio(
            label="隱藏標籤",
            options=["🎯 策略執行", "🌪️ 變盤預警", "🪐 宏觀週期指標", "🔭 空間視覺", "📈 Coinbase排名"],
            label_visibility="collapsed" 
        )
        
        st.divider()
        
        # 系統管理按鈕區 (改為更簡潔的佈局)
        st.markdown("### ⚙️ 數據管理")
        if st.button("🔄 立即同步最新數據", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # with st.expander("📥 數據回填工具", expanded=False):
        #     c1, c2 = st.columns(2)
        #     c1.button("📺 YT", key="yt_btn", use_container_width=True)
        #     c2.button("📖 Wiki", key="wiki_btn", use_container_width=True)

        st.caption(f"⚖️: F&G {FEAR_GREED_WEIGHT} | YT {YOUTUBE_WEIGHT} | Wiki {WIKIPEDIA_WEIGHT}")

     

    # 5. 全域戰情室 Header (永遠置頂)
    st.title("📊 BTC Cycle Risk Dashboard")
    
    h1, h2, h3 = st.columns([1, 1, 2])
    with h1:
        price_color = "#00FFC2" if price_pct >= 0 else "#ff5252"
        st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {price_color};">
                <div class="metric-label">BTC 當前價格 (Price)</div>
                <div class="metric-value" style="color: {price_color} !important;">${curr_price:,.0f}</div>
                <div class="metric-sub" style="color: {price_color};">
                    {"▲" if price_pct >= 0 else "▼"} {abs(price_pct):.2f}% (24h)
                </div>
                <div style="height: 25px;"></div> </div>
        """, unsafe_allow_html=True)
    with h2:
        if curr_risk <= 0.4: bar_color = "#00FFC2"
        elif curr_risk >= 0.8: bar_color = "#ff5252"
        else: bar_color = "#ffeb3b"
        
        # 確保轉換為 0-100 的數字
        p_val = int(float(curr_risk) * 100)

        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">社交風險指數 (Social Risk)</div>
                <div class="metric-value" style="color: {bar_color} !important;">{curr_risk:.2f}</div>
                <div class="risk-progress-container">
                    <div class="risk-progress-fill" style="width: {p_val}%; background-color: {bar_color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with h3:
       st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {advice_color};">
            <div class="metric-label">目前操作建議 (Current Advice)</div>
            <div class="metric-value" style="color: {advice_color} !important; font-size: 45px;">
                {advice_title} </div>
            <div class="metric-sub" style="color: {advice_color};">基於社交情緒模型計算</div>
            <div style="height: 25px;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # 6. 模組內容切換
    if menu == "🎯 策略執行":
        # 原封不動保留你的 core_sub_tabs
        core_sub_tabs = st.tabs(["📖 邏輯", "📈 風險圖表", "💰 回測", "📋 數據"])

        with core_sub_tabs[0]:
            # 這裡貼入你原本的 Markdown 說明...
            st.subheader("🎯 情緒驅動型動態 DCA 策略")
            st.markdown("""
            ### 💡 核心哲學：買在無人問津時
            本策略遵循**第一性原理**中的「市場心理學」。比特幣的牛熊週期本質上是人類情緒（貪婪與恐懼）的極端擺盪。
            當大眾瘋狂討論時，風險最高；當社交熱度冰封時，機會最大。

            ### 🏗️ 社交風險組成 (100% 權重)
            您的動態 DCA 分數完全基於以下社交維度的加權計算：
            
            1. **😱 恐懼與貪婪指數 (50%)**: 
               反映市場參與者的直接情緒壓力。
            2. **📺 YouTube 綜合熱度 (30%)**: 
               監控 KOL 與大眾影音內容的產出與觀看量，捕捉散戶進場的實質動作。
            3. **📖 Wikipedia 瀏覽量 (20%)**: 
               最純粹的「場外關注度」指標。當新手開始搜尋 "Bitcoin" 時，往往代表行情進入中後期。

            ---

            ### 🚦 社交風險操作指南
            | 風險指數 | 社交狀態 | 操作策略 | 心理狀態 |
            | :--- | :--- | :--- | :--- |
            | **0.0 - 0.2** | 絕望 / 乏人問津 | **大額抄底** | 只有你在買，別人都說比特幣死了 |
            | **0.2 - 0.4** | 冷清 / 低度關注 | **穩定買入** | 市場安靜，適合慢慢累積籌碼 |
            | **0.4 - 0.6** | 中性震盪 | **持有觀望** | 社交熱度一般，跟隨趨勢 |
            | **0.6 - 0.8** | 興奮 / FOMO 醞釀 | **停止買入** | 開始在社交媒體頻繁看到暴富新聞 |
            | **0.8 - 1.0** | 極度瘋狂 | **分批止盈** | 連街邊都在討論時，就是離場訊號 |

            > **策略重點：** > 本系統不預測價格走勢，而是透過「量化群眾瘋狂程度」來決定你的持倉力道。
            > **社交風險越高 = 獲利了結；社交風險越低 = 加大投入。**
            """)
            st.info("💡 目前您的設定將『社交情緒』作為唯一的 DCA 決策基準，其他技術指標僅供趨勢對比。")
        with core_sub_tabs[1]:
            use_log = st.checkbox("使用對數座標 (Log Scale)", value=True, key="lab_log")
            df_recent = df[df['open_time'] >= "2017-08-17"]
            st.plotly_chart(get_risk_chart_figure(df_recent, use_log=use_log), use_container_width=True)
        with core_sub_tabs[2]:
            # --- 步驟 1：參數設定 (保持原本的 Expander) ---
            st.markdown('<div class="force-light"></div>', unsafe_allow_html=True)
            with st.expander("🛠️ 步驟 1：配置投資組合與參數", expanded=True):
                col_a1, col_a2, col_a3, col_a4 = st.columns([1.5, 1.5, 1.2, 1.2])
                with col_a1:
                    total_buy_usdt = st.number_input("每日投入總預算 (USDT)", min_value=10, value=100, step=10)
                with col_a2:
                    start_date_sim = st.date_input("模擬起始日期", datetime(2020, 1, 1), key="sim_start")
                with col_a3:
                    fee_rate_pct = st.number_input("手續費 (%)", min_value=0.0, value=0.1, step=0.01)
                    fee_rate = fee_rate_pct / 100.0
                with col_a4:
                    sell_pct_val = st.number_input("每次止盈比例 (%)", min_value=0.1, max_value=100.0, value=1.0, step=1.0) / 100.0

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    buy_range = st.slider("買入風險區間 (BTC Risk)", 0.0, 1.0, (0.0, 0.4), step=0.05)
                with col_b2:
                    sell_range = st.slider("止盈風險區間 (BTC Risk)", 0.0, 1.0, (0.8, 1.0), step=0.05)

                selected_assets = st.multiselect("選擇要分配的幣種", TOP_COINS + BACKUP_COINS, default=["BTC"])
                
                asset_weights = {}
                if selected_assets:
                    st.markdown("##### ⚖️ 權重分配 (%)")
                    weight_cols = st.columns(len(selected_assets))
                    total_pct = 0
                    for i, asset in enumerate(selected_assets):
                        default_w = round(100.0 / len(selected_assets), 1)
                        w = weight_cols[i].number_input(f"{asset}", 0.0, 100.0, default_w, 5.0, key=f"lab_w_{asset}")
                        asset_weights[asset] = w / 100.0
                        total_pct += w
                    
                    if abs(total_pct - 100) > 0.1:
                        st.error(f"❌ 權重總和需為 100% (目前: {total_pct}%)")
                        st.stop()
            st.markdown('</div>', unsafe_allow_html=True) 

            # --- 步驟 2：執行運算與呈現指標 ---
            if selected_assets:
                with st.spinner("🚀 正在執行同步回測模擬..."):
                    # 重要：這裡要接收 5 個值
                    portfolio_df, asset_results, all_prices, buy_count, sell_count = run_portfolio_backtest(
                        df, asset_weights, total_buy_usdt, 
                        buy_range[0], buy_range[1],
                        sell_pct_val, sell_range[0], sell_range[1],
                        start_date_sim, fee_rate
                    )
                
                if not portfolio_df.empty:
                    # --- 【關鍵點】在此處先行計算所有變數，避免 NameError ---
                    final_p = portfolio_df.iloc[-1]
                    total_invested = final_p['Total_Cost']
                    total_equity = final_p['Equity']
                    total_net_profit = total_equity - total_invested
                    total_roi = (total_net_profit / total_invested * 100) if total_invested > 0 else 0
                    total_realized = final_p['Realized']
                    total_unrealized = total_net_profit - total_realized
                    total_fees = sum(r['fees'] for r in asset_results.values())
                    
                    # 計算 MDD
                    portfolio_df['Peak'] = portfolio_df['Equity'].cummax()
                    portfolio_df['DD'] = (portfolio_df['Equity'] - portfolio_df['Peak']) / portfolio_df['Peak']
                    mdd_val = portfolio_df['DD'].min() * 100

                    st.markdown("---")
                    st.markdown("### 🚀 策略核心指標")
                    
                    # --- UI 改為 5 欄 ---
                    roi_color = "#00e676" if total_net_profit >= 0 else "#ff5252"
                    c1, c2, c3, c4, c5 = st.columns(5)

                    with c1:
                        st.markdown(f"""<div class="metric-card" style="border-top: 4px solid {roi_color};">
                            <div class="metric-label">總獲利</div>
                            <div class="metric-value" style="color: {roi_color} !important;">${total_net_profit:,.0f}</div>
                            <div class="metric-sub">{total_roi:.1f}% ROI</div>
                        </div>""", unsafe_allow_html=True)

                    with c2:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">買入天數</div>
                            <div class="metric-value" style="color: #00e676 !important;">{buy_count} 天</div>
                            <div class="metric-sub">觸發 Buy 區間</div>
                        </div>""", unsafe_allow_html=True)

                    with c3:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">止盈天數</div>
                            <div class="metric-value" style="color: #ff5252 !important;">{sell_count} 天</div>
                            <div class="metric-sub">觸發 Sell 區間</div>
                        </div>""", unsafe_allow_html=True)

                    with c4:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">最大回撤</div>
                            <div class="metric-value" style="color: #ffb300 !important;">{mdd_val:.1f}%</div>
                            <div class="metric-sub">MDD</div>
                        </div>""", unsafe_allow_html=True)

                    with c5:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">總權益</div>
                            <div class="metric-value">${total_equity:,.0f}</div>
                            <div class="metric-sub">含市值與現金</div>
                        </div>""", unsafe_allow_html=True)

                    # --- 資金與成本指標 ---
                    st.subheader("💰 資金與成本細節")
                    r2_1, r2_2, r2_3, r2_4 = st.columns(4)
                    r2_1.metric("總投入本金", f"${total_invested:,.0f}")
                    r2_2.metric("手續費總支出", f"${total_fees:,.0f}")
                    r2_3.metric("已實現損益", f"${total_realized:,.0f}")
                    r2_4.metric("未實現損益", f"${total_unrealized:,.0f}")

                    # --- 繪圖 (加入唯一 Key) ---
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=portfolio_df['Date'], y=portfolio_df['Equity'], name='組合淨值', line=dict(color='#00e676', width=2)))
                    fig_p.add_trace(go.Scatter(x=portfolio_df['Date'], y=portfolio_df['Total_Cost'], name='投入本金', line=dict(color='#ef5350', dash='dash')))
                    fig_p.update_layout(template="plotly_dark", height=400, hovermode="x unified")
                    st.plotly_chart(fig_p, use_container_width=True, key="unique_backtest_equity_chart")

                    st.write("") # 留白
                    # ==========================================
                    # 新增：當前資產市值分佈 (圓餅圖)
                    # ==========================================
                    import plotly.express as px
                    
                    pie_data = []
                    for asset in selected_assets:
                        res = asset_results[asset]
                        curr_p = all_prices[asset].iloc[-1]
                        mkt_val = res['balance'] * curr_p
                        if mkt_val > 0:
                            pie_data.append({"幣種": asset, "當前市值": round(mkt_val, 2)})
                    
                    if pie_data:
                        st.markdown("#### 🍩 當前資產配置比例 (Market Value Allocation)")
                        df_pie = pd.DataFrame(pie_data)
                        fig_pie = px.pie(
                            df_pie, values='當前市值', names='幣種',
                            hole=0.4, # 變成環狀圖比較美
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_pie.update_layout(
                            template="plotly_dark", 
                            height=350,
                            margin=dict(l=20, r=20, t=30, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True, key="backtest_pie_chart")
                    # ==========================================
                    # 3. 單幣種拆解 (穿透分析 - 完整九大指標)
                    # ==========================================
                    st.divider()
                    st.subheader("🪙 資產組成詳情 (Individual Asset Breakdown)")
                    

                    for asset in selected_assets:
                        res = asset_results[asset]
                        curr_p = all_prices[asset].iloc[-1]
                        mkt_val = res['balance'] * curr_p
                        
                        # 核心計算 (保留原邏輯)
                        a_total_pnl = mkt_val + res['realized_pnl'] - res['cum_invested']
                        a_roi = (a_total_pnl / res['cum_invested'] * 100) if res['cum_invested'] > 0 else 0
                        a_mdd = res['mdd'] * 100
                        a_avg_cost = res['current_cost'] / res['balance'] if res['balance'] > 0 else 0
                        a_unrealized = mkt_val - res['current_cost']

                        # --- 定義摺疊標題 (在外層就能看到重點) ---
                        expander_title = f"📊 {asset} 詳情分析 ｜ ROI: {a_roi:+.2f}% ｜ 市值: ${mkt_val:,.0f}"
                        
                        st.markdown('<div class="force-light"></div>', unsafe_allow_html=True)
                        # 使用 with st.expander 進行摺疊，內部包含 9 個指標
                        with st.expander(expander_title, expanded=False):
                            # 第一區塊：策略績效 (4 個指標)
                            st.markdown("#### 📈 策略績效")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("總獲利 (Total PnL)", f"${a_total_pnl:,.2f}", f"{a_roi:.1f}% ROI")
                            c2.metric("最大回撤 (MDD)", f"{a_mdd:.2f}%", delta_color="inverse")
                            c3.metric("持倉均價 (Avg Cost)", f"${a_avg_cost:,.2f}")
                            c4.metric(f"目前 {asset} 幣價", f"${curr_p:,.2f}")

                            st.write("") # 增加間距

                            # 第二區塊：持倉與成本 (5 個指標)
                            st.markdown("#### 💰 持倉與成本")
                            c2_1, c2_2, c2_3, c2_4, c2_5 = st.columns(5)
                            c2_1.metric("當前持倉市值", f"${mkt_val:,.2f}")
                            c2_2.metric("總投入本金", f"${res['cum_invested']:,.2f}")
                            c2_3.metric("手續費總支出", f"${res['fees']:,.2f}")
                            c2_4.metric("已實現損益", f"${res['realized_pnl']:,.2f}")
                            c2_5.metric("未實現損益", f"${a_unrealized:,.2f}")
                            
                            st.caption(f"目前持倉: {res['balance']:.4f} {asset} | 分配權重: {asset_weights[asset]*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True) # 關閉容器    
                else:
                    st.error("❌ 無交易紀錄。")
            pass
        # [子分頁 3] 詳細數據
        with core_sub_tabs[3]:
            cols = ['open_time', 'close', 'total_risk', 'fear_greed', 'wiki_views', 'youtube_val', 'blockchain_active', 'coinglass_funding', 'news_count', 'obituary_count', 'cmc_rank']
            display_cols = [c for c in cols if c in df.columns]
            rename_map = {
                'open_time': '日期', 'close': 'BTC價格', 'total_risk': '社交風險',
                'fear_greed': 'F&G', 'wiki_views': 'Wiki瀏覽', 'youtube_val': 'YT綜合熱度',
                'blockchain_active': '鏈上地址', 'coinglass_funding': '資金費率',
                'news_count': '新聞數', 'obituary_count': '死亡宣告', 'cmc_rank': 'CMC排名'
            }

            # 4. 準備數據：移除 .tail(14)，改為顯示全部，並按日期降序排列
            df_full_display = df[display_cols].sort_values('open_time', ascending=False)
            
            # 5. 格式化顯示
            st.dataframe(
                df_full_display.rename(columns=rename_map), 
                use_container_width=True, 
                height=800, # 增加表格高度方便滾動
                hide_index=True # 隱藏 Streamlit 預設索引
            )
            
            # 6. 提供 CSV 下載按鈕 (這對全歷史數據非常有用)
            csv_data = df_full_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下載完整 CSV 數據報告",
                data=csv_data,
                file_name=f"btc_full_history_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            
            pass
    elif menu == "🌪️ 變盤預警":
        # 建立兩個子分頁
        adv_sub_tabs = st.tabs(["📉 BBW 變盤預警", "💸 MFI 資金流向", "🔍 歷史相似度", "⚡ 歷史波動率", "🚀 TTM Squeeze", "📈 Fisher Transform","🎢 Coppock Curve"])
        # [子分頁 1] BBW 變盤預警
        with adv_sub_tabs[0]:
            st.subheader("🌪️ 布林通道寬度 (BBW) - 變盤預警系統")
            st.markdown("""
            **如何解讀圖表上的菱形訊號？**
            當 BBW 跌破閾值時，圖表會自動標示出「歷史變盤點」。
            - **🟢 綠色菱形**：發生在低風險區 (Risk < 0.4)，歷史上常為 **起漲點**。
            - **🔴 紅色菱形**：發生在高風險區 (Risk > 0.6)，歷史上常為 **崩盤前夕**。
            - **🟡 黃色菱形**：發生在中性區，方向未明，需搭配其他指標。
            
            *試著調整下方的「壓縮閾值」，看看能不能過濾出歷史上的大行情！*
            """)
            
            col_bb1, col_bb2 = st.columns([1, 1])
            with col_bb1:
                # 預設閾值調低一點點到 0.05，這樣訊號比較精準
                bbw_threshold = st.slider("BBW 壓縮閾值 (Squeeze Threshold)", 0.01, 0.15, 0.05, 0.005)
            with col_bb2:
                bb_period = st.number_input("布林通道週期 (預設 20)", value=20, step=1)

            # 呼叫函數
            fig_bbw, current_bbw = get_bbw_squeeze_chart(df, lookback=bb_period, squeeze_threshold=bbw_threshold)
            
            # 顯示當前狀態指標
            c1, c2, c3 = st.columns(3)
            c1.metric("當前 BBW 數值", f"{current_bbw:.4f}")
            
            is_squeezing = current_bbw < bbw_threshold
            status_text = "🔥 變盤蓄力中 (SQUEEZING)" if is_squeezing else "💨 波動正常"
            status_color = "inverse" if is_squeezing else "normal"
            c2.metric("狀態判定", status_text, delta="注意方向" if is_squeezing else None, delta_color=status_color)
            
            # 顯示方向預測
            if is_squeezing:
                if curr_risk < 0.4:
                    bias = "🚀 看漲暴發 (Bullish)"
                elif curr_risk > 0.6:
                    bias = "📉 看跌崩盤 (Bearish)"
                else:
                    bias = "⚖️ 中性突破 (Neutral)"
                c3.metric("結合風險指標預判", bias)
            else:
                c3.metric("結合風險指標預判", "--")

            st.plotly_chart(fig_bbw, use_container_width=True)

        # [2-2] MFI 資金流向 (Smart Money Flow)
        with adv_sub_tabs[1]:
            st.subheader("💸 MFI 資金流向 (Smart Money Flow)")
            st.markdown("""
            **MFI (Money Flow Index) 是「成交量加權」的 RSI，能反映市場的「真金白銀」流向。**
            
            這張圖表幫你判斷市場的「油箱」還剩多少油：
            - **🔴 紅色點 (>80)**：**買力耗盡 (Buyer Exhaustion)**。散戶資金都已經買進去了，場外沒有新錢能推升價格，通常是**階段性頂部**。
            - **🟢 綠色點 (<20)**：**賣壓衰竭 (Seller Exhaustion)**。想賣的人都賣光了，成交量急劇萎縮，通常是**聰明錢吸籌的底部**。
            
            *進階用法：當幣價創新高，但下方 MFI 曲線卻越來越低（沒有碰到紅線），這是典型的頂背離。*
            """)
            
            mfi_len = st.slider("MFI 週期", 7, 30, 26, help="預設 14。數值越小反應越快。")
            
            fig_mfi, curr_mfi = get_mfi_divergence_chart(df, period=mfi_len)
            
            # 儀表板
            c1, c2 = st.columns(2)
            c1.metric("當前 MFI 強度", f"{curr_mfi:.1f}")
            
            if curr_mfi > 80:
                s_text = "🔥 過熱：買力耗盡"
                s_col = "inverse"
            elif curr_mfi < 20:
                s_text = "💎 冰點：賣壓衰竭"
                s_col = "normal"
            else:
                s_text = "💨 中性：流動正常"
                s_col = "off"
                
            c2.metric("資金狀態", s_text, delta_color=s_col)

            st.plotly_chart(fig_mfi, use_container_width=True)
        # [2-3] 歷史相似度
        with adv_sub_tabs[2]:
            st.subheader("🔍 歷史分形相似度搜索 (Fractal Similarity)")
            st.markdown("AI 自動比對當前的 **「風險指標結構」** 與 **「價格型態」**，在歷史數據中尋找最相似的片段（Top Matches），並展示該片段隨後的走勢。")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                lookback = st.slider("回溯比對天數 (Lookback)", 30, 180, 90, step=10)
            with col_s2:
                forecast_d = st.slider("觀察後續天數 (Forecast)", 10, 90, 60, step=5)

            if st.button("🚀 開始比對分析"):
                with st.spinner("正在掃描歷史數據庫..."):
                    top_matches, curr_price_arr, curr_risk_arr = find_similar_patterns(df, lookback_days=lookback, forecast_days=forecast_d)
                
                if not top_matches:
                    st.warning("沒有找到相關性 > 80% 的歷史片段，請嘗試調整回溯天數。")
                else:
                    # 統計數據
                    avg_return = np.mean([m['future_return'] for m in top_matches]) * 100
                    positive_moves = sum(1 for m in top_matches if m['future_return'] > 0)
                    win_rate = (positive_moves / len(top_matches)) * 100
                    
                    # 顯示摘要結果
                    m1, m2, m3 = st.columns(3)
                    m1.metric("找到相似片段", f"{len(top_matches)} 組")
                    m2.metric("歷史後續平均漲跌", f"{avg_return:.2f}%", delta_color="normal")
                    m3.metric("歷史上漲機率", f"{win_rate:.0f}%")
                    
                    st.divider()
                    
                    # 繪圖
                    for i, match in enumerate(top_matches):
                        st.markdown(f"### 🔗 Match #{i+1}: {match['start_date'].strftime('%Y-%m-%d')} ~ {match['end_date'].strftime('%Y-%m-%d')}")
                        st.caption(f"相似度得分: {match['score']*100:.1f}% | 後續 {forecast_d} 天漲跌: {match['future_return']*100:.2f}%")
                        
                        fig_sim = go.Figure()
                        
                        # 1. 為了畫在一起，我們需要建立一個統一的 X 軸 (0 ~ lookback + forecast)
                        x_axis = list(range(lookback + forecast_d))
                        
                        # 2. 處理數據 (歸一化以對齊起點)
                        # Current Price (只有前 lookback 天)
                        curr_p_norm = (curr_price_arr - curr_price_arr.min()) / (curr_price_arr.max() - curr_price_arr.min())
                        
                        # Historical Price (包含前 lookback + 後 forecast)
                        full_hist_price = np.concatenate([match['hist_price_data'], match['future_price_data']])
                        # 針對 hist 的前段做 min-max 縮放，後段跟隨比例
                        h_min = match['hist_price_data'].min()
                        h_max = match['hist_price_data'].max()
                        hist_p_norm = (full_hist_price - h_min) / (h_max - h_min)

                        # 3. 畫圖：當前走勢 (粗白線)
                        fig_sim.add_trace(go.Scatter(
                            x=list(range(lookback)), 
                            y=curr_p_norm, 
                            name="Current Trend (Now)",
                            line=dict(color='white', width=4)
                        ))
                        
                        # 4. 畫圖：歷史走勢 (前半段實線，後半段虛線)
                        # 歷史-比對段
                        fig_sim.add_trace(go.Scatter(
                            x=list(range(lookback)), 
                            y=hist_p_norm[:lookback], 
                            name="Historical Match",
                            line=dict(color='#00e5ff', width=2)
                        ))
                        # 歷史-未來段
                        color_future = '#4caf50' if match['future_return'] > 0 else '#ff5252'
                        fig_sim.add_trace(go.Scatter(
                            x=list(range(lookback, lookback + forecast_d)), 
                            y=hist_p_norm[lookback:], 
                            name="What Happened Next",
                            line=dict(color=color_future, width=2, dash='dot')
                        ))
                        
                        # 分隔線
                        fig_sim.add_vline(x=lookback-1, line_width=1, line_dash="dash", line_color="gray")
                        fig_sim.add_annotation(x=lookback-1, y=0, text="現在", showarrow=False, yshift=10)

                        fig_sim.update_layout(
                            title="價格型態疊加 (歸一化視圖)",
                            height=350,
                            margin=dict(l=10, r=10, t=30, b=10),
                            template="plotly_dark",
                            xaxis=dict(showgrid=False, title="Days"),
                            yaxis=dict(showgrid=False, visible=False) # 隱藏 Y 軸數值，因為是歸一化的
                        )
                        st.plotly_chart(fig_sim, use_container_width=True)
        # [2-4] 歷史波動率 (新增)
        with adv_sub_tabs[3]:
            st.subheader("⚡ 歷史波動率 (Historical Volatility, HV)")
            st.markdown("""
            **暴風雨前的寧靜。** 當 HV 降至極低水平 (<25%) 時，代表市場正在蓄力，隨後通常會出現劇烈的單邊行情。
            - **策略**：在低波動區間 (紅色虛線下方) 佈局，等待突破。
            """)
            
            c1, c2 = st.columns(2)
            with c1: hv_window = st.slider("計算窗口 (天)", 7, 90, 30, help="越短越靈敏，越長越平滑。")
            with c2: hv_threshold = st.number_input("低波動閾值 (%)", 10, 50, 25, help="低於此值視為壓縮區。")
            
            fig_hv, curr_hv = get_historical_volatility_chart(df, window=hv_window, threshold=hv_threshold)
            
            st.metric("當前年化波動率 (HV)", f"{curr_hv:.2f}%", delta="壓縮中" if curr_hv < hv_threshold else "波動擴大", delta_color="inverse" if curr_hv < hv_threshold else "normal")
            st.plotly_chart(fig_hv, use_container_width=True)
        # [2-5] TTM Squeeze (可調參數版)
        with adv_sub_tabs[4]:
            st.subheader("🚀 TTM Squeeze (擠壓指標)")
            st.markdown("""
            **BBW 的終極進化版。**
            * **🔴 紅色鑽石 (Squeeze On)**：布林通道「縮進」了肯特納通道。波動率極度壓縮，**埋伏訊號**。
            * **🔥 綠色星星 (Fired)**：壓縮結束，行情點火噴出。
            * **📊 柱狀圖**：預判爆發方向。
            """)
            
            # --- 新增：參數調整區 (放在摺疊選單裡保持介面乾淨) ---
            with st.expander("⚙️ 進階參數設定 (調整靈敏度)", expanded=False):
                c_p1, c_p2, c_p3 = st.columns(3)
                with c_p1: 
                    ttm_len = st.number_input("計算週期 (Length)", value=50, min_value=5)
                with c_p2:
                    # 布林帶標準差 (通常固定 2.0)
                    bb_mult = st.number_input("布林帶寬度 (BB Mult)", value=2.0, step=0.1, format="%.1f")
                with c_p3:
                    # 這是關鍵！調大 = 訊號變多(容易擠壓)，調小 = 訊號變少(嚴格)
                    kc_mult = st.number_input("肯特納寬度 (KC Mult)", value=1.5, step=0.1, format="%.1f", help="標準是 1.5。比特幣波動大，若訊號太少可嘗試調大至 2.0")

            # 將參數傳入繪圖函數
            fig_ttm, is_sqz = get_ttm_squeeze_chart(df, length=ttm_len, mult=bb_mult, length_kc=ttm_len, mult_kc=kc_mult)
            
            c1, c2 = st.columns(2)
            c1.metric("當前狀態", "🔴 壓縮蓄力" if is_sqz else "🟢 波動釋放")
            c2.metric("操作建議", "準備埋伏 / 等待突破" if is_sqz else "順勢交易")
            
            st.plotly_chart(fig_ttm, use_container_width=True)
        # [2-6] Fisher Transform (新增)
        with adv_sub_tabs[5]:
            st.subheader("📈 Ehlers Fisher Transform (費雪變換)")
            st.markdown("""
            **比 RSI 更銳利的轉折指標。** 利用常態分佈原理，將價格訊號銳利化，專抓 **「極值反轉」**。
            * **🟢 藍色三角 (Buy)**：指標在極低位置 (<-1.5) 出現黃金交叉，這通常是 **V 型反轉** 的起點。
            * **🔴 紅色三角 (Sell)**：指標在極高位置 (>1.5) 出現死亡交叉，趨勢可能瞬間反轉。
            * **觀察重點**：它的線條非常直，一旦轉頭通常不回頭。
            """)
            
            with st.expander("⚙️ 參數設定", expanded=False):
                fish_len = st.number_input("計算週期 (Length)", value=180, min_value=5, help="標準為 10。越小越敏銳。")
            
            fig_fish, curr_fish = get_ehlers_fisher_chart(df, length=fish_len)
            
            c1, c2 = st.columns(2)
            c1.metric("Fisher 值", f"{curr_fish:.2f}")
            
            if curr_fish > 2.5: rec = "危險 (極高)"; c="inverse"
            elif curr_fish < -2.5: rec = "機會 (極低)"; c="normal"
            else: rec = "觀望"; c="off"
                
            c2.metric("極值狀態", rec, delta_color=c)
            
            st.plotly_chart(fig_fish, use_container_width=True)
        # [2-7] Coppock Curve (介面更新)
        with adv_sub_tabs[6]:
            st.subheader("🎢 Coppock Curve (古波克曲線) - 頂底雙抓")
            st.markdown("""
            **長線週期雷達。**
            * **🔼 粉紅箭頭 (大底)**：指標在水下深處 (Buy Zone) 拐頭向上。
            * **🔽 青色箭頭 (大頂)**：指標在高空熱區 (Sell Zone) 拐頭向下。
            * **參數建議**：如果箭頭太多，請將閾值調得更嚴格（買點更負，賣點更正）。
            """)
            
            with st.expander("⚙️ 參數與靈敏度設定", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1: wma_l = st.number_input("WMA Length", value=10)
                with c2: roc1_l = st.number_input("ROC 1 (長)", value=24)
                with c3: roc2_l = st.number_input("ROC 2 (短)", value=20)
                
                st.divider()
                
                c4, c5 = st.columns(2)
                with c4: 
                    # 買點閾值 (負數)
                    buy_thresh = st.slider("🟢 買點過濾 (Buy Threshold)", -70, 0, -60, help="數值越負，只抓深跌後的反彈。建議 -15 或 -20。")
                with c5:
                    # 賣點閾值 (正數) - 新增這個
                    sell_thresh = st.slider("🔴 賣點過濾 (Top Threshold)", 0, 100, 70, help="數值越大，只抓過熱後的崩盤。建議 20 或 30。")
            
            # 呼叫函數
            fig_cc, curr_cc = get_coppock_curve_chart(
                df, wma_len=wma_l, roc1_len=roc1_l, roc2_len=roc2_l, 
                bottom_threshold=buy_thresh, top_threshold=sell_thresh
            )
            
            # 顯示當前狀態
            c_val, c_status = st.columns(2)
            c_val.metric("當前數值", f"{curr_cc:.2f}")
            
            if curr_cc > sell_thresh:
                status_msg = "🔥 極度過熱 (Sell Zone)"
                status_color = "inverse"
            elif curr_cc < buy_thresh:
                status_msg = "❄️ 極度超跌 (Buy Zone)"
                status_color = "normal"
            else:
                status_msg = "💨 中性震盪"
                status_color = "off"
                
            c_status.metric("區間狀態", status_msg, delta_color=status_color)
            
            st.plotly_chart(fig_cc, use_container_width=True)

    elif menu == "🪐 宏觀週期指標":   
    # ------------------------------------------------------------
    # Tab 3: 宏觀週期指標 (包含原本的 Tab 7, 8, 9)
    # ------------------------------------------------------------
        macro_sub_tabs = st.tabs(["👑 週期", "🌊 梅耶倍數", "📐 冪律法則", "💎 囤幣指標", "🏭 銅金比", "🌊 全球流動性", "🔗 SPX 脫鉤分析", "📉 2M StochRSI"])
        
        # [子分頁 1] 週期大師
        with macro_sub_tabs[0]:
            st.subheader("👑 週期大師指標 (Cycle Master)")
            st.markdown("""
            **這張圖表展示了比特幣週期的「天花板」與「地板」：**
            
            1. **💀 Pi Cycle Top (紅線與橘虛線)**：
               - 歷史上，當 **橘虛線 (111SMA)** 向上穿越 **紅線 (350SMA x 2)** 時，精準標記了 2013、2017、2021 的最高點。
               - **策略：** 如果看到圖上出現 "X" 標記，建議大幅減倉或清倉。
               
            2. **💎 200 Week MA (綠線)**：
               - 這是比特幣的「絕對估值底線」。歷史上價格很難長時間低於此線。
               - **策略：** 當價格觸碰或低於綠線時，是「賣房抄底」的歷史機遇。
            """)
            
            use_log_cycle = st.checkbox("使用對數座標 (Log Scale)", value=True, key="cycle_log")
            
            # 呼叫函數
            fig_cycle = get_cycle_master_chart(df, use_log=use_log_cycle)
            st.plotly_chart(fig_cycle, use_container_width=True)

        # [子分頁 2] 冪律法則
        with macro_sub_tabs[1]:
            st.subheader("📐 比特幣冪律法則 (Bitcoin Power Law)")
            st.markdown("""
            **這不是技術分析，這是數學規律。**
            
            冪律模型發現比特幣價格與時間呈現 $Price = a * Days^{5.8}$ 的關係。
            - **藍色虛線 (Fair Value)**：比特幣的「地心引力」，價格無論怎麼飛，最後都會被拉回這條線。
            - **綠色底線**：比特幣的「物理極限底」，歷史上從未有效跌破。
            - **紅色上軌**：歷史上的泡沫極限。
            """)
            
            # 呼叫函數
            # 強制使用 Log 座標，因為冪律模型在線性座標下看不出規律
            fig_pl = get_power_law_chart(df, use_log=True)
            st.plotly_chart(fig_pl, use_container_width=True)

        # [子分頁 3] 囤幣指標
        with macro_sub_tabs[2]:
            st.subheader("💎 AHR999 囤幣指標 (Hoarding Index)")
            st.markdown("""
            **這是一個專為「定投黨」設計的指標。**
            它告訴你現在的價格相對於「長期價值」和「短期成本」是便宜還是貴。
            
            - **🟢 綠色區間 (< 0.45)**：歷史大底。建議：**加大購買力度 / 梭哈**。
            - **🔵 藍色區間 (0.45 - 1.2)**：合理區間。建議：**堅持定投**。
            - **🔴 紅色區間 (> 1.2)**：價格偏高。建議：**停止定投，持有觀望**。
            """)
            
            # 呼叫函數
            fig_ahr, current_ahr = get_ahr999_chart(df)
            
            # 顯示儀表板數字
            c1, c2, c3 = st.columns(3)
            c1.metric("AHR999 指數", f"{current_ahr:.3f}")
            
            if current_ahr < 0.45:
                rec_text = "💎 買爆 (ALL IN)"
                rec_color = "normal" # 綠色
            elif current_ahr < 1.2:
                rec_text = "👌 堅持定投 (DCA)"
                rec_color = "off" # 灰色/中性
            else:
                rec_text = "🚫 停止買入 (STOP BUY)"
                rec_color = "inverse" # 紅色
                
            c2.metric("操作建議", rec_text, delta=None, delta_color=rec_color)
            c3.metric("指標狀態", "低估" if current_ahr < 1 else "高估")

            st.plotly_chart(fig_ahr, use_container_width=True)
        # [子分頁 4] 梅耶倍數
        with macro_sub_tabs[3]:
            st.subheader("🌊 Mayer Multiple (梅耶倍數)")
            st.markdown("價格 / 200日均線。 >2.4 為瘋狂，<0.6 為超跌。")
            fig_mm, curr_mm = get_mayer_multiple_chart(df)
            c1, c2 = st.columns(2)
            c1.metric("Mayer Multiple", f"{curr_mm:.2f}")
            if curr_mm > 2.4: status="🔥 泡沫區"; c="inverse"
            elif curr_mm < 0.6: status="💎 抄底區"; c="normal"
            else: status="💨 正常區"; c="off"
            c2.metric("估值狀態", status, delta_color=c)
            st.plotly_chart(fig_mm, use_container_width=True)
        # [子分頁 5]  銅金比
        with macro_sub_tabs[4]:
            st.subheader("🏭 銅金比 (Copper/Gold Ratio) - 宏觀經濟晴雨表")
            st.markdown("""
            **公式：** `銅價 / 金價` (Log Regression Channel)
            - **銅 (Risk-On)**：代表工業需求與經濟繁榮。
            - **金 (Risk-Off)**：代表避險與恐懼。
            - **📉 觸及下軌**：宏觀大底 (Risk-Off Bottom)。
            - **📈 觸及上軌**：經濟過熱 (Risk-On Top)。
            """)
            cg_lookback = st.slider("回歸通道回測年數", 3, 20, 17, help="調整線性回歸的取樣範圍。")
            with st.spinner("正在計算宏觀數據..."):
                fig_cg, curr_cg = get_copper_gold_ratio_chart(lookback_years=cg_lookback)
            st.metric("當前銅金比指數", f"{curr_cg:.5f}")
            st.plotly_chart(fig_cg, use_container_width=True)
        # [子分頁 6] 全球流動性 (修正為 Fed Net Liquidity)
        with macro_sub_tabs[5]:
            st.subheader("🌊 全球流動性指數 (Global Liquidity Proxy)")
            st.markdown("""
            **公式：** `(Nasdaq 100 * Gold) / 10`
            這是一個華爾街常用的流動性代理指標。
            - **邏輯**：科技股代表風險偏好，黃金代表貨幣貶值。兩者同時上漲意味著法幣系統的流動性正在氾濫。
            - **藍色區域**：流動性趨勢。
            - **橘色線條**：比特幣趨勢 (已標準化對齊)。
            """)
            
            liq_lookback = st.slider("流動性回測年數", 3, 30, 5)
            
            with st.spinner("正在分析全球資金流..."):
                # 【修正點】這裡現在對應的是 Proxy 函數，只回傳 2 個變數
                fig_liq, curr_liq = get_global_liquidity_chart(lookback_years=liq_lookback)
            
            st.metric("流動性指數", f"{curr_liq:.1f}")
            st.plotly_chart(fig_liq, use_container_width=True)
        # [子分頁 7] SPX 脫鉤分析 (最新版)
        with macro_sub_tabs[6]:
            st.subheader("🔗 比特幣 vs 標普 500 季度分析儀")
            st.markdown("""
            **「當美股不再能左右比特幣時，就是大行情蓄勢待發的時刻。」**
            * **📊 90 日窗口**：這代表一個完整的日曆季度。每一根柱子都反映了過去三個月的宏觀同步度。
            * **🟢 綠色 (r < 0.2)**：**獨立行情**。比特幣展現非相關性，可能是避險資金湧入或場內週期發動。
            * **🔴 紅色 (r > 0.2)**：**聯動行情**。比特幣被視為風險資產，受美股漲跌制約。
            """)
            
            with st.spinner("正在對齊全球宏觀數據..."):
                fig_spx, spx_stats = get_btc_spx_decoupling_chart(df, window=90)
            
            if fig_spx:
                # 修改點 2：將欄位改為 4 欄，騰出空間放「目前脫鉤」
                c1, c2, c3, c4 = st.columns(4)
                
                c1.metric("當前相關性 (90D)", f"{spx_stats['current']:.2f}")
                
                # 顯示目前脫鉤天數
                streak_val = spx_stats['current_streak']
                if spx_stats['is_decoupled']:
                    c2.metric("目前脫鉤狀態", f"第 {streak_val} 天", delta="獨立走勢中", delta_color="normal")
                else:
                    c2.metric("目前脫鉤狀態", "聯動中", delta="跟隨美股", delta_color="inverse")
                
                c3.metric("歷史平均脫鉤", f"{spx_stats['avg']:.1f} 天")
                c4.metric("歷史最長脫鉤", f"{spx_stats['max']:.0f} 天")
                
                st.plotly_chart(fig_spx, use_container_width=True)
            else:
                st.error("無法加載標普 500 數據，請稍後再試。")
        # [子分頁 8] 雙月線 Stoch RSI 
        with macro_sub_tabs[7]:
            st.subheader("📉 雙月線 Stoch RSI (宏觀頂底雷達)")
            st.markdown("""
            **「當你拉長週期，雜訊就會消失。」**
            透過將 K 線週期拉長至 2 個月，我們能避開中短期的插針與洗盤。
            - **🟢 歷史大底 (Buy)**：指標進入綠色超賣區 (<30) 後，快線向上穿越慢線形成**黃金交叉**。
            - **🔴 歷史大頂 (Sell)**：指標進入紅色超買區 (>70) 後，快線向下穿越慢線形成**死亡交叉**。
            """)
            
            with st.spinner("正在計算 2M Stoch RSI 宏觀級別..."):
                fig_stoch, k_val, d_val = get_2m_stoch_rsi_chart(df)
            
            # 建立儀表板
            c1, c2, c3 = st.columns(3)
            c1.metric("當前 %K 線 (快線)", f"{k_val:.1f}")
            c2.metric("當前 %D 線 (慢線)", f"{d_val:.1f}")
            
            if k_val < 30:
                s_text = "超賣區 (大底醞釀)"
                s_color = "normal"
            elif k_val > 70:
                s_text = "超買區 (大頂風險)"
                s_color = "inverse"
            else:
                s_text = "中性區間"
                s_color = "off"
            c3.metric("宏觀位階", s_text, delta_color=s_color)
            
            st.plotly_chart(fig_stoch, use_container_width=True)
    elif menu == "🔭 空間視覺":
    # ------------------------------------------------------------
    # Tab 4: 3D 視覺化 (包含原本的 Tab 4)
    # ------------------------------------------------------------
        st.subheader("🧊 3D 沉浸式分析")
        
        # 模式選擇
        viz_mode = st.radio(
            "選擇分析模式：",
            ["線性歷史 (Linear History)", "週期疊加 (Cycle Overlay)"],
            help="線性歷史：沿著時間軸飛行，看完整歷史。\n週期疊加：將四次減半週期疊在一起，比較波型相似度。",
            horizontal=True
        )

        if viz_mode == "線性歷史 (Linear History)":
            st.markdown("**說明：** 沿著時間長河飛行，觀察長期趨勢與成交量變化 (從 2017 Binance 數據開始)。")
            
            # =========================================================
            # 【修改開始】這裡加入過濾邏輯，只取 2017-08-17 之後的數據
            # =========================================================
            cutoff_date = pd.to_datetime("2017-08-17")
            # 建立一個新的 df_linear，只包含 Binance 時期以後的資料
            df_linear = df[df['open_time'] >= cutoff_date].reset_index(drop=True)
            
            # 將過濾後的 df_linear 傳入生成函數
            html_data = generate_interactive_html(df_linear) 
            # =========================================================
            
            file_name = "btc_linear_3d.html"
            
        else:
            # 週期疊加模式：繼續使用完整的 df (包含 2012-2017)，這樣才能畫出舊週期
            st.markdown("**說明：** 這是 **尋找分形 (Fractals)** 的神器。不同週期的走勢像千層蛋糕一樣疊在一起...")
            html_data = generate_overlay_html(df) 
            file_name = "btc_cycle_overlay_3d.html"

        # 下載按鈕
        st.download_button(
            label=f"🎮 下載 {viz_mode} (.html)",
            data=html_data,
            file_name=file_name,
            mime="text/html"
        )
        
        st.components.v1.html(html_data, height=500, scrolling=False)
        st.caption("👆 上方為預覽視窗 (按一下可試玩，但建議下載後全螢幕體驗最佳)")
    elif menu == "📈 Coinbase排名":
        st.subheader("📈 Coinbase 實時排名 (散戶心理指標)")
        
        st.markdown("""
        > 💡 **左側核心邏輯：** 觀察零售端「增量資金」的極限。
        > 歷史證明，當 Coinbase 登頂美國區總榜 **第 1 名** 之日，往往就是 **牛市週期大頂** (如 2017/12、2021/4、2021/11)，誤差通常在 7-10 天內。
        """)
        
        with st.spinner("正在同步 App Store 數據..."):
            data = fetch_coinbase_rank()

        if data:
            rank = data['overall']
            finance_rank = data['finance']
            top_100 = data['top_apps']
            
            # 判斷顏色與狀態
            if rank:
                overall_text = f"#{rank}"
                color = "#ff1744" if rank <= 10 else "#ff9100"
                status = "🔥 極度狂熱 (逃頂區)" if rank <= 10 else "📈 散戶進場中"
            else:
                # 總榜抓不到時，顯示類別排名作為參考
                overall_text = "100名外 (冷清)"
                color = "#00e5ff"
                status = "😴 市場冷靜 (定投區)"

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {color};">
                    <div class="metric-label">App Store 總榜 (Overall)</div>
                    <div class="metric-value" style="color: {color} !important;">{overall_text}</div>
                </div>""", unsafe_allow_html=True)
            
            with c2:
                f_rank_val = f"#{finance_rank}" if finance_rank else "未入榜"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">財務類別精確排名 (Finance)</div>
                    <div class="metric-value" style="color: #ffffff !important;">{f_rank_val}</div>
                </div>""", unsafe_allow_html=True)

            # 顯示列表
            st.divider()
            st.markdown(f"#### 🏆 美國區總榜 Top 100 (判定：{status})")
            with st.container(height=600):
                for i, name in enumerate(top_100):
                    is_crypto = any(x in name.lower() for x in ["coinbase", "binance", "crypto.com", "kraken"])
                    is_ai = any(x in name.lower() for x in ["chatgpt", "gemini", "ai ", "grok"])
                    
                    marker, l_color = ("💰", "#ff1744") if is_crypto else (("🤖", "#00FFC2") if is_ai else ("📱", "gray"))
                    st.markdown(f"**{i+1}.** {marker} <span style='color:{l_color};'>{name}</span>", unsafe_allow_html=True)
        else:
            st.error("無法獲取數據，請確認 API 限制。")


if __name__ == "__main__":
    main()
