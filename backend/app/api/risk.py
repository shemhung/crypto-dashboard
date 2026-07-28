import pandas as pd
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.risk import (
    RiskCalculationRequest,
    RiskCalculationResponse,
    RiskPointResponse,
)
from backend.app.services.risk_service import compute_risk


router = APIRouter(
    prefix="/api/v1/risk",
    tags=["risk"],
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