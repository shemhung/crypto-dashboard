Crypto Risk Dashboard

以 BTC為主的的全端系統。專案將原本的單體 Streamlit 應用重構為 React 前端、FastAPI 後端、外部 PostgreSQL 資料庫與自動化資料更新流程，並使用 Docker Compose 統一建置與啟動。

本專案僅供技術研究與系統開發展示，不構成任何投資建議。

系統架構

flowchart LR
    U[使用者瀏覽器] -->|HTTP :5173| N[Nginx]
   
    N --> R[React + TypeScript]
    
    N -->|/api/*| F[FastAPI + Uvicorn]
  
    F -->|SQLAlchemy| P[(PostgreSQL / Supabase)]
    F --> B[Binance Market API]
    F --> G[Fear & Greed API]
    A[GitHub Actions] --> W[Wikipedia / YouTube Update]
    A --> J[Daily Risk Update Job]
    W --> P
    J --> P

目前 Docker Compose 管理兩個 application services：

frontend: React production build + Nginx
backend:  FastAPI + Uvicorn

資料庫不在 Docker Compose 內，後端會透過 DATABASE_URL 連線到已存在的 PostgreSQL／Supabase。


快速啟動：Docker Compose

1. Clone 專案

git clone --branch refactor/backend-core --single-branch \
  https://github.com/shemhung/crypto-dashboard.git

cd crypto-dashboard

2. 建立環境變數

cp .env.example .env

至少設定：

DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DATABASE

每日資料更新還會使用：

YOUTUBE_API_KEY=your_youtube_api_key

請勿將真正的 .env、API Key 或資料庫密碼提交到 GitHub。

3. 建置並啟動

docker compose up --build -d

4. 開啟服務

Frontend: http://localhost:5173

FastAPI: http://localhost:8000

Swagger UI: http://localhost:8000/docs

Health API: http://localhost:8000/api/v1/health

Data Status API: http://localhost:8000/api/v1/data-status

5. 查看狀態與紀錄

docker compose ps
docker compose logs -f

6. 停止服務

docker compose down

功能特色

Dashboard：顯示 BTC 最新價格、總風險、價格風險、社群風險與近期紀錄。

Risk Analysis：以歷史價格與風險資料呈現市場週期，支援不同時間區間查詢。

Portfolio Backtest：設定多資產權重、每日投入金額、買賣風險區間、賣出比例與手續費，計算 ROI、最大回撤、權益曲線及交易紀錄。

Data Status：檢查資料庫、Binance 市場 API 與 BTCUSDT 風險資料的新鮮度。

Daily Data Update：透過 GitHub Actions 更新 Wikipedia、YouTube、BTC 市場資料與 Risk Score。

Containerized Deployment：React production build 由 Nginx 提供，FastAPI 由 Uvicorn 執行，兩者透過 Docker Compose 管理。
