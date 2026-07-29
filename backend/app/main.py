from fastapi import FastAPI

from backend.app.api.backtest import router as backtest_router
from backend.app.api.health import router as health_router
from backend.app.api.risk import router as risk_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Crypto Dashboard API",
    description="BTC 市場資料、風險分析與回測 API",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    health_router,
    prefix="/api/v1",
)

app.include_router(risk_router)
app.include_router(backtest_router)