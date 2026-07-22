from dataclasses import dataclass


@dataclass(frozen=True)
class RiskWeights:
    """風險模型使用的權重設定。"""

    # 總風險權重
    social: float = 1.0
    price: float = 0.0
    derivative: float = 0.0
    volume: float = 0.0

    # 社交風險內部權重
    fear_greed: float = 0.5
    youtube: float = 0.3
    wikipedia: float = 0.2
    blockchain: float = 0.0
    coinglass: float = 0.0
    google_news: float = 0.0
    obituaries: float = 0.0
    cmc_trending: float = 0.0

    # 尚未使用的資料來源
    google_trends: float = 0.0
    reddit: float = 0.0
    twitter: float = 0.0
    cryptopanic: float = 0.0
    bitinfocharts: float = 0.0
    lunarcrush: float = 0.0


DEFAULT_RISK_WEIGHTS = RiskWeights()