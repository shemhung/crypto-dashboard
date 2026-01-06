# -*- coding: utf-8 -*-
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import sys

# ==========================================
# 1. 設定區 (安全版)
# ==========================================
# Google Sheet 網址
SHEET_URL = "https://docs.google.com/spreadsheets/d/1eDMd7hOd5CCj6TpDvMSGiA5YsEASZ3he9cX9sKaB18g"

# 本地金鑰檔案名稱
JSON_KEYFILE = 'service_account.json'

def get_api_key():
    """
    安全獲取 API Key 的邏輯：
    1. 先嘗試讀取系統環境變數 (GitHub Actions 用)
    2. 如果沒有，嘗試讀取本地 .streamlit/secrets.toml (本地開發用)
    3. 絕對不可以在這裡寫死 Key！
    """
    # 1. 嘗試從環境變數獲取 (GitHub Actions)
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if api_key:
        return api_key.strip()
    
    # 2. 嘗試從本地 secrets.toml 獲取 (本地電腦)
    try:
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "youtube_api_key" in line and "=" in line:
                        # 簡單解析：找到 youtube_api_key = "..."
                        return line.split("=")[1].strip().strip('"').strip("'")
    except Exception as e:
        print(f"⚠️ 讀取本地 Secrets 失敗: {e}")
        pass
    
    return None

# 獲取 Key
YOUTUBE_API_KEY = get_api_key()

# 檢查是否有拿到 Key，沒有就報錯，防止程式空轉
if not YOUTUBE_API_KEY:
    print("❌ 嚴重錯誤：找不到 YOUTUBE_API_KEY！")
    print("   - 如果在本地：請確認 .streamlit/secrets.toml 裡面有設定")
    print("   - 如果在 GitHub：請確認 Settings -> Secrets -> Actions 裡面有設定")
    sys.exit(1) # 強制停止程式

# ==========================================
# 2. 連線 Google Sheets (支援 本地/GitHub 雙模式)
# ==========================================
def connect_gsheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    
    # 判斷是否在 GitHub Actions 環境 (透過檢查環境變數)
    if "GCP_SERVICE_ACCOUNT_JSON" in os.environ:
        print("🤖 檢測到雲端環境，使用環境變數憑證...")
        try:
            # 將 JSON 字串轉為字典
            creds_dict = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except Exception as e:
            print(f"❌ 環境變數憑證解析失敗: {e}")
            return None
    else:
        print("💻 檢測到本地環境，使用 service_account.json...")
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(JSON_KEYFILE, scope)
        except Exception as e:
            print(f"❌ 找不到本地憑證檔案: {e}")
            return None

    client = gspread.authorize(creds)
    sh = client.open_by_url(SHEET_URL)
    return sh

# ==========================================
# 3. 抓取 YouTube 數據核心邏輯
# ==========================================
def fetch_youtube_monthly_data(year, month, keywords=['Bitcoin', 'BTC']):
    # 計算月份起訖時間
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    
    # RFC 3339 格式
    start_str = start_date.strftime('%Y-%m-%dT00:00:00Z')
    end_str = end_date.strftime('%Y-%m-%dT23:59:59Z')
    
    print(f"  正在搜尋 {year}-{month:02d} ... ", end='')
    
    # 1. 搜尋影片
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        'key': YOUTUBE_API_KEY,
        'q': ' OR '.join(keywords),
        'part': 'snippet',
        'type': 'video',
        'maxResults': 50, # 每次分析前50部熱門影片
        'publishedAfter': start_str,
        'publishedBefore': end_str,
        'order': 'relevance'
    }
    
    try:
        resp = requests.get(search_url, params=search_params)
        if resp.status_code != 200:
            print(f"❌ API Error {resp.status_code}")
            return None
            
        data = resp.json()
        videos = data.get('items', [])
        video_count = len(videos)
        
        if not videos:
            print("⚠️ 無影片")
            return None
        
        # 2. 獲取影片詳細統計 (觀看數)
        video_ids = [v['id']['videoId'] for v in videos if 'videoId' in v['id']]
        stats_url = "https://www.googleapis.com/youtube/v3/videos"
        stats_params = {
            'key': YOUTUBE_API_KEY,
            'id': ','.join(video_ids),
            'part': 'statistics'
        }
        
        stats_resp = requests.get(stats_url, params=stats_params)
        views_list = []
        
        if stats_resp.status_code == 200:
            for item in stats_resp.json().get('items', []):
                try:
                    views = int(item['statistics'].get('viewCount', 0))
                    views_list.append(views)
                except:
                    pass
        
        # 3. 計算指標
        total_views = sum(views_list)
        avg_views = np.mean(views_list) if views_list else 0
        # 高流量影片佔比 (>10萬觀看)
        high_view_ratio = sum(1 for v in views_list if v > 100000) / len(views_list) if views_list else 0
        
        print(f"✅ 分析 {video_count} 支影片，平均觀看 {int(avg_views)}")
        
        return {
            'date': start_date.strftime('%Y-%m-%d'),
            'video_count': video_count,
            'total_views': total_views,
            'avg_views': avg_views,
            'high_view_ratio': high_view_ratio
        }
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        return None

# ==========================================
# 4. 主控流程：更新 Sheet
# ==========================================
def update_youtube_sheet(months_back=1):
    print("=" * 60)
    print("📺 開始回填 YouTube -> Google Sheets (yt_data)")
    print("=" * 60)

    # 連線 Sheet
    try:
        sh = connect_gsheet()
        if not sh: return
        
        # 嘗試開啟 yt_data 分頁，沒有就建立
        try:
            ws = sh.worksheet("yt_data")
        except:
            print("⚠️ 找不到 'yt_data' 分頁，正在建立...")
            ws = sh.add_worksheet(title="yt_data", rows="1000", cols="10")

        # 讀取現有數據
        existing_data = ws.get_all_records()
        existing_df = pd.DataFrame(existing_data)
        
        # 處理空 Sheet 的情況
        if existing_df.empty:
            existing_df = pd.DataFrame(columns=['date', 'video_count', 'total_views', 'avg_views', 'high_view_ratio'])
            print("ℹ️  Sheet 為空，將建立新數據。")
        else:
            # 確保 date 欄位是字串格式以便比對
            if 'date' in existing_df.columns:
                existing_df['date'] = existing_df['date'].astype(str)
            print(f"✓ 讀取到 {len(existing_df)} 筆現有數據。")

    except Exception as e:
        print(f"❌ Google Sheet 連線或讀取失敗: {e}")
        return

    # 計算需要抓取的月份
    today = datetime.now()
    new_rows = []
    
    # 從本月往回推 N 個月
    for i in range(months_back):
        d = today - timedelta(days=i*30)
        year, month = d.year, d.month
        target_date_str = datetime(year, month, 1).strftime('%Y-%m-%d')
        
        # 檢查是否已存在 (如果是當月，強制更新；如果是歷史月份且已存在，則跳過)
        is_current_month = (year == today.year and month == today.month)
        
        if not is_current_month and not existing_df.empty and target_date_str in existing_df['date'].values:
            print(f"  ⏭️  {target_date_str} 歷史數據已存在，跳過。")
            continue
            
        # 執行抓取
        result = fetch_youtube_monthly_data(year, month)
        if result:
            new_rows.append(result)
        
        # 避免 API Rate Limit
        time.sleep(1) 

    # 如果有新數據，進行合併與上傳
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        
        # 轉換日期格式確保一致
        new_df['date'] = new_df['date'].astype(str)
        
        # 如果是當月數據，先刪除舊的當月數據（因為要更新）
        if not existing_df.empty:
            # 過濾掉新數據中已有的日期 (即覆蓋邏輯)
            existing_df = existing_df[~existing_df['date'].isin(new_df['date'])]
            final_df = pd.concat([existing_df, new_df])
        else:
            final_df = new_df
            
        # 排序
        final_df = final_df.sort_values('date').drop_duplicates(subset=['date'])
        
        # 處理 NaN 值 (轉為 0 或空字串，否則 JSON 上傳會錯)
        final_df = final_df.fillna(0)
        
        print("📤 正在上傳至 Google Sheets...")
        try:
            # 準備上傳資料 (包含標題)
            data_to_upload = [final_df.columns.values.tolist()] + final_df.values.tolist()
            
            # 清空並寫入
            ws.clear()
            ws.update(data_to_upload)
            print(f"🎉 成功！已更新 {len(new_rows)} 筆數據，目前共 {len(final_df)} 筆。")
        except Exception as e:
            print(f"❌ 上傳失敗: {e}")
    else:
        print("✅ 數據已是最新，無需更新。")

# ==========================================
# 5. 程式進入點 (支援自動化參數)
# ==========================================
if __name__ == "__main__":
    # 如果有命令列參數 (例如: python youtube_backfill.py 1)
    if len(sys.argv) > 1:
        try:
            months = int(sys.argv[1])
            update_youtube_sheet(months)
        except ValueError:
            print("❌ 參數錯誤，請輸入整數月份。")
    else:
        # 本地執行時，詢問使用者
        try:
            print("\n請選擇模式：")
            print("1. 自動更新 (預設，更新最近 1 個月)")
            print("2. 深度回填 (輸入月份數)")
            choice = input("請輸入 (直接按 Enter 跑預設): ").strip()
            
            if not choice:
                update_youtube_sheet(1)
            elif choice.isdigit():
                update_youtube_sheet(int(choice))
            else:
                update_youtube_sheet(1)
        except KeyboardInterrupt:
            print("\n已取消。")