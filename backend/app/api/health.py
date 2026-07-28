from fastapi import APIRouter


router = APIRouter(
    tags=["health"],
)


@router.get("/health")
def health_check() -> dict[str, str]:
    """確認後端服務是否正常運作。"""

    return {
        "status": "ok",
        "service": "crypto-dashboard-api",
    }