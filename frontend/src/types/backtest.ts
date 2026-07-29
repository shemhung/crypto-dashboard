export interface BacktestMarketPoint {
  open_time: string;
  asset_price: number;
  total_risk: number;
}


export interface BacktestRequest {
  records: BacktestMarketPoint[];

  buy_amount: number;
  buy_min: number;
  buy_max: number;

  sell_pct: number;
  sell_min: number;
  sell_max: number;

  start_date: string;
  fee_rate: number;
}


export interface BacktestTrade {
  date: string;
  type: "BUY" | "SELL";
  price: number;
  risk: number;
  value_usdt: number;
  amount: number;
  fee: number;
  balance: number;
}


export interface BacktestPortfolioPoint {
  date: string;
  equity: number;
  invested: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_fees: number;
  avg_cost: number;
  peak_equity: number;
}


export interface BacktestResponse {
  trade_count: number;
  buy_days: number;
  sell_days: number;
  final_price: number;
  trades: BacktestTrade[];
  portfolio: BacktestPortfolioPoint[];
}

export interface PortfolioAllocation {
  asset: string;
  weight: number;
}


export interface PortfolioBacktestRequest {
  risk_symbol: string;

  start_date: string;
  end_date: string;

  daily_budget: number;

  buy_min: number;
  buy_max: number;

  sell_min: number;
  sell_max: number;

  sell_pct: number;
  fee_rate: number;

  allocations: PortfolioAllocation[];
}


export interface PortfolioBacktestSummary {
  total_contributed: number;
  total_equity: number;
  total_profit: number;

  roi_pct: number;
  mdd_pct: number;

  buy_days: number;
  sell_days: number;
  trade_count: number;

  cash_balance: number;
  market_value: number;

  total_fees: number;
  realized_pnl: number;
  unrealized_pnl: number;
}


export interface PortfolioEquityPoint {
  date: string;

  cash: number;
  market_value: number;
  equity: number;
  contributed: number;

  realized_pnl: number;
  unrealized_pnl: number;

  total_fees: number;
  peak_equity: number;
  drawdown_pct: number;
}


export interface PortfolioAssetResult {
  asset: string;
  weight: number;

  balance: number;
  last_price: number;

  market_value: number;
  cost_basis: number;
  contributed: number;

  realized_pnl: number;
  unrealized_pnl: number;

  profit: number;
  roi_pct: number;
  fees: number;
}


export interface PortfolioTrade {
  date: string;

  asset: string;
  type: "BUY" | "SELL";

  price: number;
  risk: number;
  amount: number;

  value_usdt: number;
  fee: number;
}


export interface PortfolioBacktestResponse {
  summary: PortfolioBacktestSummary;

  equity_curve: PortfolioEquityPoint[];

  assets: PortfolioAssetResult[];

  trades: PortfolioTrade[];
}