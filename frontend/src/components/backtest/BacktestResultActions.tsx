import type {
  PortfolioBacktestRequest,
  PortfolioBacktestResponse,
} from "../../types/backtest";

import {
  downloadCsv,
} from "../../utils/csv";

import "./BacktestResultActions.css";


interface BacktestResultActionsProps {
  request: PortfolioBacktestRequest;
  result: PortfolioBacktestResponse;
}


function formatUsd(
  value: number,
): string {
  return new Intl.NumberFormat(
    "zh-TW",
    {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 2,
    },
  ).format(value);
}


function formatRiskRange(
  minimum: number,
  maximum: number,
): string {
  return [
    minimum.toFixed(2),
    maximum.toFixed(2),
  ].join(" ～ ");
}


function formatPercentageFromRatio(
  value: number,
): string {
  return `${(value * 100).toFixed(2)}%`;
}


function BacktestResultActions({
  request,
  result,
}: BacktestResultActionsProps) {
  const filenamePrefix = [
    "portfolio-backtest",
    request.start_date,
    request.end_date,
  ].join("-");


  function exportSummary() {
    const rows = [
      [
        "開始日期",
        request.start_date,
      ],
      [
        "結束日期",
        request.end_date,
      ],
      [
        "每日投入預算 USDT",
        request.daily_budget,
      ],
      [
        "買入風險下限",
        request.buy_min,
      ],
      [
        "買入風險上限",
        request.buy_max,
      ],
      [
        "止盈風險下限",
        request.sell_min,
      ],
      [
        "止盈風險上限",
        request.sell_max,
      ],
      [
        "每次止盈比例 %",
        request.sell_pct * 100,
      ],
      [
        "交易手續費率 %",
        request.fee_rate * 100,
      ],
      [
        "累積投入 USDT",
        result.summary.total_contributed,
      ],
      [
        "總權益 USDT",
        result.summary.total_equity,
      ],
      [
        "總獲利 USDT",
        result.summary.total_profit,
      ],
      [
        "ROI %",
        result.summary.roi_pct,
      ],
      [
        "最大回撤 %",
        result.summary.mdd_pct,
      ],
      [
        "現金餘額 USDT",
        result.summary.cash_balance,
      ],
      [
        "持倉市值 USDT",
        result.summary.market_value,
      ],
      [
        "總手續費 USDT",
        result.summary.total_fees,
      ],
      [
        "已實現損益 USDT",
        result.summary.realized_pnl,
      ],
      [
        "未實現損益 USDT",
        result.summary.unrealized_pnl,
      ],
      [
        "買入天數",
        result.summary.buy_days,
      ],
      [
        "止盈天數",
        result.summary.sell_days,
      ],
      [
        "交易筆數",
        result.summary.trade_count,
      ],
    ];

    downloadCsv(
      `${filenamePrefix}-summary.csv`,
      [
        "項目",
        "數值",
      ],
      rows,
    );
  }


  function exportAssets() {
    const rows = result.assets.map(
      (asset) => [
        asset.asset,
        asset.weight * 100,
        asset.balance,
        asset.last_price,
        asset.market_value,
        asset.cost_basis,
        asset.contributed,
        asset.realized_pnl,
        asset.unrealized_pnl,
        asset.profit,
        asset.roi_pct,
        asset.fees,
      ],
    );

    downloadCsv(
      `${filenamePrefix}-assets.csv`,
      [
        "幣種",
        "目標權重 %",
        "持有數量",
        "期末價格 USDT",
        "目前市值 USDT",
        "剩餘成本基礎 USDT",
        "累積投入 USDT",
        "已實現損益 USDT",
        "未實現損益 USDT",
        "總損益 USDT",
        "ROI %",
        "手續費 USDT",
      ],
      rows,
    );
  }


  function exportEquityCurve() {
    const rows =
      result.equity_curve.map(
        (point) => [
          point.date,
          point.cash,
          point.market_value,
          point.equity,
          point.contributed,
          point.realized_pnl,
          point.unrealized_pnl,
          point.total_fees,
          point.peak_equity,
          point.drawdown_pct,
        ],
      );

    downloadCsv(
      `${filenamePrefix}-equity.csv`,
      [
        "日期",
        "現金餘額 USDT",
        "持倉市值 USDT",
        "組合總權益 USDT",
        "累積投入 USDT",
        "已實現損益 USDT",
        "未實現損益 USDT",
        "累積手續費 USDT",
        "歷史最高權益 USDT",
        "回撤 %",
      ],
      rows,
    );
  }


  function exportTrades() {
    const rows = result.trades.map(
      (trade) => [
        trade.date,
        trade.asset,
        trade.type,
        trade.price,
        trade.risk,
        trade.risk * 100,
        trade.amount,
        trade.value_usdt,
        trade.fee,
      ],
    );

    downloadCsv(
      `${filenamePrefix}-trades.csv`,
      [
        "日期",
        "幣種",
        "交易類型",
        "成交價格 USDT",
        "風險值",
        "風險百分比 %",
        "交易數量",
        "交易金額 USDT",
        "手續費 USDT",
      ],
      rows,
    );
  }


  return (
    <section className="backtest-result-actions">
      <div className="backtest-result-actions__heading">
        <div>
          <span>Strategy Summary</span>

          <h3>本次回測參數</h3>
        </div>

        <div className="backtest-result-actions__downloads">
          <button
            type="button"
            onClick={exportSummary}
          >
            匯出摘要 CSV
          </button>

          <button
            type="button"
            onClick={exportAssets}
          >
            匯出幣種績效
          </button>

          <button
            type="button"
            onClick={exportEquityCurve}
          >
            匯出淨值曲線
          </button>

          <button
            type="button"
            onClick={exportTrades}
            disabled={
              result.trades.length === 0
            }
          >
            匯出交易紀錄
          </button>
        </div>
      </div>


      <div className="backtest-parameter-grid">
        <article>
          <span>回測期間</span>

          <strong>
            {request.start_date}
            {" → "}
            {request.end_date}
          </strong>
        </article>

        <article>
          <span>每日投入預算</span>

          <strong>
            {formatUsd(
              request.daily_budget,
            )}
          </strong>
        </article>

        <article>
          <span>買入風險區間</span>

          <strong>
            {formatRiskRange(
              request.buy_min,
              request.buy_max,
            )}
          </strong>
        </article>

        <article>
          <span>止盈風險區間</span>

          <strong>
            {formatRiskRange(
              request.sell_min,
              request.sell_max,
            )}
          </strong>
        </article>

        <article>
          <span>每次止盈比例</span>

          <strong>
            {formatPercentageFromRatio(
              request.sell_pct,
            )}
          </strong>
        </article>

        <article>
          <span>交易手續費</span>

          <strong>
            {formatPercentageFromRatio(
              request.fee_rate,
            )}
          </strong>
        </article>
      </div>


      <div className="backtest-allocation-summary">
        <span>目標配置</span>

        <div>
          {request.allocations.map(
            (allocation) => (
              <span
                key={allocation.asset}
                className={
                  "backtest-allocation-chip"
                }
              >
                <strong>
                  {allocation.asset}
                </strong>

                {formatPercentageFromRatio(
                  allocation.weight,
                )}
              </span>
            ),
          )}
        </div>
      </div>
    </section>
  );
}


export default BacktestResultActions;