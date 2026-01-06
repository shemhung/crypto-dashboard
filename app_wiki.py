# -*- coding: utf-8 -*-
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import sys

# ==========================================
# 1. 設定區
# ==========================================
# Google Sheet 網址 (請確認網址正確)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1eDMd7hOd5CCj6TpDvMSGiA5YsEASZ3he9cX9sKaB18g"

# 本地金鑰檔案名稱
JSON_KEYFILE = 'service_account.json'

# ==========================================
# 2. 連線 Google Sheets (支援 本地/GitHub 雙模式)
# ==========================================
def connect_gsheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    
    # 判斷是否在 GitHub Actions 環境
    if "GCP_SERVICE_ACCOUNT_JSON" in os.environ:
        print("🤖 檢測到雲端環境，使用環境變數憑證...")
        try:
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
# 3. 抓取 Wiki 數據主邏輯
# ==========================================
def fetch_wiki_history():
    print("=" * 60)
    print("📚 開始回填 Wikipedia (Bitcoin) -> Google Sheets")
    print("=" * 60)

    # 1. 連線 Google Sheet
    try:
        sh = connect_gsheet()
        if not sh: return

        # 嘗試開啟 wiki_data 分頁，沒有就建立
        try:
            ws = sh.worksheet("wiki_data")
        except:
            print("⚠️ 找不到 'wiki_data' 分頁，正在建立...")
            ws = sh.add_worksheet(title="wiki_data", rows="1000", cols="5")

        existing_data = ws.get_all_records()
        existing_df = pd.DataFrame(existing_data)
        
        # 決定開始日期
        if not existing_df.empty and 'date_wiki' in existing_df.columns:
            # 確保轉成 datetime
            existing_df['date_wiki'] = pd.to_datetime(existing_df['date_wiki'])
            last_date = existing_df['date_wiki'].max()
            start_date = last_date + timedelta(days=1)
            print(f"✓ 讀取到 {len(existing_df)} 筆數據，最後日期: {last_date.date()}")
            print(f"  接續從 {start_date.date()} 開始抓取...")
        else:
            print("ℹ️  Sheet 為空或無有效數據，執行全量抓取 (從 2015 年開始)...")
            start_date = datetime(2015, 7, 1)
            existing_df = pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Google Sheet 連線失敗: {e}")
        return

    end_date = datetime.now()
    
    # 如果已經是最新的，就不跑了
    if start_date >= end_date:
        print("✅ 數據已是最新，無需更新。")
        return

    headers = {
        'User-Agent': 'BitcoinRiskBot/1.0 (Personal Education Project)'
    }

    all_new_data = []
    
    # 2. 分段抓取 (每次一年)
    fetch_ptr = start_date
    
    while fetch_ptr < end_date:
        chunk_end = fetch_ptr + timedelta(days=365)
        if chunk_end > end_date:
            chunk_end = end_date
        
        start_str = fetch_ptr.strftime('%Y%m%d')
        end_str = chunk_end.strftime('%Y%m%d')
        
        print(f"  抓取區間: {start_str} - {end_str} ... ", end="", flush=True)
        
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Bitcoin/daily/{start_str}/{end_str}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                items = data.get('items', [])
                
                count = 0
                for item in items:
                    raw_date = item['timestamp']
                    # 轉成 YYYY-MM-DD
                    date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                    views = item['views']
                    
                    all_new_data.append({
                        'date_wiki': date_str,
                        'wiki_views': views
                    })
                    count += 1
                print(f"✅ ({count} 筆)")
            else:
                print(f"❌ HTTP {resp.status_code}")
                
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            
        fetch_ptr = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    # 3. 上傳回 Google Sheet
    if all_new_data:
        new_df = pd.DataFrame(all_new_data)
        
        # 合併舊資料
        if not existing_df.empty:
            # 統一日期格式為字串，方便上傳
            existing_df['date_wiki'] = existing_df['date_wiki'].dt.strftime('%Y-%m-%d')
            final_df = pd.concat([existing_df, new_df])
        else:
            final_df = new_df
            
        # 去重與排序
        final_df = final_df.drop_duplicates(subset=['date_wiki']).sort_values('date_wiki')
        
        # 處理 NaN
        final_df = final_df.fillna(0)
        
        print("📤 正在上傳至 Google Sheets...")
        try:
            # gspread 需要將 DataFrame 轉為 list of lists，並包含標題
            data_to_upload = [final_df.columns.values.tolist()] + final_df.values.tolist()
            
            ws.clear() # 清空舊的
            ws.update(data_to_upload) # 寫入新的
            
            print(f"🎉 回填完成！總共 {len(final_df)} 筆數據。")
            
        except Exception as e:
            print(f"❌ 上傳失敗: {e}")
    else:
        print("⚠️ 本次沒有抓取到新數據。")

if __name__ == "__main__":
    fetch_wiki_history()