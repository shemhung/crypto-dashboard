from fastapi import FastAPI

from backend.app.api.health import router as health_router
from backend.app.api.risk import router as risk_router


app = FastAPI(
    title="Crypto Dashboard API",
    description="BTC 市場資料、風險分析與回測 API",
    version="0.1.0",
)


app.include_router(health_router)
app.include_router(risk_router)