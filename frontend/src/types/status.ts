export type SystemStatus =
  | "healthy"
  | "degraded"
  | "unavailable";


export interface DatabaseStatus {
  connected: boolean;
}


export interface RiskDataStatus {
  symbol: string;

  record_count: number;

  earliest_time: string | null;
  latest_time: string | null;

  age_hours: number | null;

  is_stale: boolean;
}


export interface MarketApiStatus {
  provider: string;
  available: boolean;
}


export interface DataStatusResponse {
  status: SystemStatus;

  checked_at: string;

  database: DatabaseStatus;

  risk_data: RiskDataStatus;

  market_api: MarketApiStatus;
}