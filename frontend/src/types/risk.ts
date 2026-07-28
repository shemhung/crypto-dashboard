export interface StoredRiskPoint {
  symbol: string;
  score_time: string;
  price: number | null;
  total_risk: number;
  price_risk: number;
  social_risk: number;
  risk_level: string;
}

export interface RiskHistoryResponse {
  count: number;
  records: StoredRiskPoint[];
}