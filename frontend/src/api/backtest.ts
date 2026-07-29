import { apiPost } from "./client";

import type {
  BacktestRequest,
  BacktestResponse,
  PortfolioBacktestRequest,
  PortfolioBacktestResponse,
} from "../types/backtest";


/**
 * 原本的單一幣種回測。
 */
export function createBacktest(
  request: BacktestRequest,
  signal?: AbortSignal,
): Promise<BacktestResponse> {
  return apiPost<
    BacktestResponse,
    BacktestRequest
  >(
    "/api/v1/backtests",
    request,
    signal,
  );
}


/**
 * 多幣種投資組合回測。
 */
export function createPortfolioBacktest(
  request: PortfolioBacktestRequest,
  signal?: AbortSignal,
): Promise<PortfolioBacktestResponse> {
  return apiPost<
    PortfolioBacktestResponse,
    PortfolioBacktestRequest
  >(
    "/api/v1/backtests/portfolio",
    request,
    signal,
  );
}