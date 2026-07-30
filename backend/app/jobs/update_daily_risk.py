from __future__ import annotations

from backend.app.core.database import (
    get_engine,
)
from backend.app.repositories.risk_update_repository import (
    RiskUpdateRepository,
)
from backend.app.services.risk_update_service import (
    update_daily_risk,
)


def main() -> int:
    print(
        "Starting daily BTC risk update..."
    )

    try:
        repository = RiskUpdateRepository(
            engine=get_engine()
        )

        result = update_daily_risk(
            repository=repository,
            symbol="BTCUSDT",
        )

    except Exception as exc:
        print(
            "[ERROR] Daily risk update failed:"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    print(
        "Daily risk update completed."
    )
    print(
        f"Symbol: {result.symbol}"
    )
    print(
        "Latest score time: "
        f"{result.latest_score_time}"
    )
    print(
        "Latest price: "
        f"{result.latest_price:,.2f}"
    )
    print(
        "Latest total risk: "
        f"{result.latest_total_risk:.4f}"
    )
    print(
        "Market rows upserted: "
        f"{result.market_rows}"
    )
    print(
        "Risk rows upserted: "
        f"{result.risk_rows}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())