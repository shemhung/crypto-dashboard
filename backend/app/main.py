from fastapi import FastAPI


app = FastAPI(
    title="Crypto Dashboard API",
    description="BTC 市場資料、風險分析與回測 API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """確認後端服務是否正常運作。"""

    return {
        "status": "ok",
        "service": "crypto-dashboard-api",
    }
