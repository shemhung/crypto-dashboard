from backend.app.services.risk_service import compute_rainbow_risk


def test_compute_rainbow_risk_stays_between_zero_and_one():
    test_prices = [
        0.01,
        100.0,
        10_000.0,
        100_000.0,
        1_000_000.0,
    ]

    for price in test_prices:
        result = compute_rainbow_risk(price)

        assert 0.0 <= result <= 1.0