import { apiGet } from "./client";

import type {
  RiskHistoryResponse,
  StoredRiskPoint,
} from "../types/risk";


export function getLatestRisk(
  symbol = "BTCUSDT",
  signal?: AbortSignal,
): Promise<StoredRiskPoint> {
  const encodedSymbol = encodeURIComponent(symbol);

  return apiGet<StoredRiskPoint>(
    `/api/v1/risk/latest?symbol=${encodedSymbol}`,
    signal,
  );
}


export function getRiskHistory(
  symbol = "BTCUSDT",
  limit = 30,
  signal?: AbortSignal,
): Promise<RiskHistoryResponse> {
  const encodedSymbol = encodeURIComponent(symbol);

  return apiGet<RiskHistoryResponse>(
    `/api/v1/risk/history?symbol=${encodedSymbol}&limit=${limit}`,
    signal,
  );
}