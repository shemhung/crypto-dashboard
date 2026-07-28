import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from backend.app.schemas.risk import (
    RiskCalculationRequest,
    RiskCalculationResponse,
    RiskHistoryResponse,
    RiskPointResponse,
    StoredRiskPointResponse,
)
from backend.app.services.risk_service import compute_risk
from sqlalchemy.exc import SQLAlchemyError

from backend.app.dependencies import get_market_repository
from backend.app.repositories.market_repository import (
    MarketRepository,
)


    
router = APIRouter(
    prefix="/api/v1/risk",
    tags=["risk"],
)

@router.get(
    "/latest",
    response_model=StoredRiskPointResponse,
)
def get_latest_risk(
    symbol: str = Query(
        default="BTCUSDT",
        min_length=3,
        max_length=20,
    ),
    repository: MarketRepository = Depends(
        get_market_repository
    ),
) -> StoredRiskPointResponse:
    """取得指定資產最新一筆風險資料。"""

    try:
        record = repository.get_latest_risk(
            symbol=symbol.upper(),
        )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="資料庫目前無法使用",
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="找不到指定資產的風險資料",
        )

    return StoredRiskPointResponse(**record)


@router.get(
    "/history",
    response_model=RiskHistoryResponse,
)
def get_risk_history(
    symbol: str = Query(
        default="BTCUSDT",
        min_length=3,
        max_length=20,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
    repository: MarketRepository = Depends(
        get_market_repository
    ),
) -> RiskHistoryResponse:
    """取得指定資產的歷史風險資料。"""

    try:
        records = repository.get_risk_history(
            symbol=symbol.upper(),
            limit=limit,
        )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="資料庫目前無法使用",
        ) from exc

    response_records = [
        StoredRiskPointResponse(**record)
        for record in records
    ]

    return RiskHistoryResponse(
        count=len(response_records),
        records=response_records,
    )



@router.post(
    "/calculate",
    response_model=RiskCalculationResponse,
)
def calculate_risk(
    request: RiskCalculationRequest,
) -> RiskCalculationResponse:
    """
    根據傳入的市場與社群資料計算風險。

    此 API 不會呼叫外部 API 或資料庫。
    """

    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="records 不可為空",
        )

    market_data = pd.DataFrame(
        [
            record.model_dump()
            for record in request.records
        ]
    )

    try:
        result = compute_risk(market_data)

    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"風險資料格式錯誤：{exc}",
        ) from exc

    required_columns = {
        "open_time",
        "close",
        "price_risk",
        "social_risk",
        "total_risk",
    }

    missing_columns = required_columns - set(result.columns)

    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "風險服務未產生必要欄位："
                f"{sorted(missing_columns)}"
            ),
        )

    response_records = []

    for _, row in result.iterrows():
        response_records.append(
            RiskPointResponse(
                open_time=pd.Timestamp(
                    row["open_time"]
                ).to_pydatetime(),
                close=float(row["close"]),
                price_risk=float(row["price_risk"]),
                social_risk=float(row["social_risk"]),
                total_risk=float(row["total_risk"]),
            )
        )

    return RiskCalculationResponse(
        count=len(response_records),
        records=response_records,
    )