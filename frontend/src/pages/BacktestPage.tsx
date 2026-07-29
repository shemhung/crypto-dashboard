import {
  type FormEvent,
  useState,
} from "react";

import {
  createPortfolioBacktest,
} from "../api/backtest";

import RiskRangeSlider, {
  type RiskRangeValue,
} from "../components/forms/RiskRangeSlider";

import PortfolioEquityChart from "../components/charts/PortfolioEquityChart";
import AllocationChart from "../components/charts/AllocationChart";
import PortfolioTradeTable from "../components/tables/PortfolioTradeTable";
import type {
  PortfolioBacktestRequest,
  PortfolioBacktestResponse,
} from "../types/backtest";

import "./BacktestPage.css";


const HISTORY_START_DATE = "2017-08-17";


const SUPPORTED_ASSETS = [
  "BTC",
  "ETH",
  "LINK",
  "FET",
  "RENDER",
  "DOGE",
  "LTC",
  "WLD",
  "BNB",
  "TRX",
  "ADA",
  "ALGO",
  "ATOM",
  "DASH",
  "XTZ",
  "IOTA",
  "XRP",
  "SOL",
  "BCH",
  "XLM",
  "AAVE",
  "ETC",
  "FIL",
  "QNT",
] as const;


type SupportedAsset =
  (typeof SUPPORTED_ASSETS)[number];


type AssetWeights = Partial<
  Record<SupportedAsset, number>
>;


function formatLocalDate(
  date: Date,
): string {
  const year = date.getFullYear();

  const month = String(
    date.getMonth() + 1,
  ).padStart(2, "0");

  const day = String(
    date.getDate(),
  ).padStart(2, "0");

  return `${year}-${month}-${day}`;
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


function formatMetricPercent(
  value: number,
): string {
  return `${value.toFixed(2)}%`;
}


function formatWeight(
  value: number,
): string {
  return `${(value * 100).toFixed(1)}%`;
}


/**
 * 幣種選擇改變時，
 * 自動將權重平均分配到 100%。
 */
function createEqualWeights(
  assets: SupportedAsset[],
): AssetWeights {
  if (assets.length === 0) {
    return {};
  }

  const result: AssetWeights = {};

  const baseWeight = Math.floor(
    (100 / assets.length) * 100,
  ) / 100;

  let remainingWeight = 100;

  assets.forEach(
    (asset, index) => {
      const isLast =
        index === assets.length - 1;

      const weight = isLast
        ? Number(
            remainingWeight.toFixed(2),
          )
        : baseWeight;

      result[asset] = weight;

      remainingWeight -= weight;
    },
  );

  return result;
}


function BacktestPage() {
  const today = formatLocalDate(
    new Date(),
  );

  const [startDate, setStartDate] =
    useState("2020-01-01");

  const [endDate, setEndDate] =
    useState(today);

  const [dailyBudget, setDailyBudget] =
    useState(100);

  /*
   * 前端顯示百分比：
   * 1 代表 1%
   *
   * 送到 API 時再除以 100。
   */
  const [
    sellPercentage,
    setSellPercentage,
  ] = useState(1);

  const [
    feePercentage,
    setFeePercentage,
  ] = useState(0.1);

  const [
    buyRange,
    setBuyRange,
  ] = useState<RiskRangeValue>([
    0,
    0.4,
  ]);

  const [
    sellRange,
    setSellRange,
  ] = useState<RiskRangeValue>([
    0.8,
    1,
  ]);

  const [
    selectedAssets,
    setSelectedAssets,
  ] = useState<SupportedAsset[]>([
    "BTC",
    "ETH",
  ]);

  const [weights, setWeights] =
    useState<AssetWeights>({
      BTC: 60,
      ETH: 40,
    });

  const [result, setResult] =
    useState<PortfolioBacktestResponse | null>(
      null,
    );

  const [running, setRunning] =
    useState(false);

  const [error, setError] =
    useState("");


  const weightTotal =
    selectedAssets.reduce(
      (total, asset) => {
        return total + (
          weights[asset] ?? 0
        );
      },
      0,
    );


  function handleAssetToggle(
    asset: SupportedAsset,
  ) {
    const alreadySelected =
      selectedAssets.includes(asset);

    const nextAssets =
      alreadySelected
        ? selectedAssets.filter(
            (selectedAsset) =>
              selectedAsset !== asset,
          )
        : [
            ...selectedAssets,
            asset,
          ];

    if (nextAssets.length === 0) {
      setError(
        "至少必須選擇一個幣種。",
      );

      return;
    }

    setSelectedAssets(nextAssets);

    /*
     * 幣種增加或刪除時，
     * 重新平均分配權重。
     */
    setWeights(
      createEqualWeights(nextAssets),
    );

    setResult(null);
    setError("");
  }


  function updateWeight(
    asset: SupportedAsset,
    value: number,
  ) {
    setWeights(
      (currentWeights) => ({
        ...currentWeights,
        [asset]: value,
      }),
    );

    setResult(null);
  }


  function validateForm(): string {
    if (
      !startDate ||
      !endDate
    ) {
      return "請選擇開始日期與結束日期。";
    }

    if (
      startDate < HISTORY_START_DATE
    ) {
      return (
        `開始日期不可早於 ` +
        `${HISTORY_START_DATE}。`
      );
    }

    if (startDate > endDate) {
      return "開始日期不可晚於結束日期。";
    }

    if (endDate > today) {
      return "結束日期不可晚於今天。";
    }

    if (dailyBudget <= 0) {
      return "每日投入金額必須大於 0。";
    }

    if (
      buyRange[0] >= buyRange[1]
    ) {
      return "買入風險下限必須小於上限。";
    }

    if (
      sellRange[0] >= sellRange[1]
    ) {
      return "止盈風險下限必須小於上限。";
    }

    if (
      buyRange[1] > sellRange[0]
    ) {
      return "買入與止盈風險區間不可重疊。";
    }

    if (
      sellPercentage <= 0 ||
      sellPercentage > 100
    ) {
      return (
        "每次止盈比例必須大於 0%，" +
        "且不可超過 100%。"
      );
    }

    if (
      feePercentage < 0 ||
      feePercentage > 10
    ) {
      return (
        "手續費率必須介於 " +
        "0% 與 10% 之間。"
      );
    }

    if (selectedAssets.length === 0) {
      return "至少必須選擇一個幣種。";
    }

    const hasInvalidWeight =
      selectedAssets.some(
        (asset) => {
          const weight =
            weights[asset] ?? 0;

          return (
            weight <= 0 ||
            weight > 100
          );
        },
      );

    if (hasInvalidWeight) {
      return (
        "每個幣種的權重必須大於 0%，" +
        "且不可超過 100%。"
      );
    }

    if (
      Math.abs(weightTotal - 100) >
      0.01
    ) {
      return (
        "幣種權重總和必須等於 100%。" +
        `目前為 ${weightTotal.toFixed(2)}%。`
      );
    }

    return "";
  }


  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    const validationError =
      validateForm();

    if (validationError) {
      setError(validationError);
      return;
    }

    const controller =
      new AbortController();

    const request:
      PortfolioBacktestRequest = {
        risk_symbol: "BTCUSDT",

        start_date: startDate,
        end_date: endDate,

        daily_budget: dailyBudget,

        buy_min: buyRange[0],
        buy_max: buyRange[1],

        sell_min: sellRange[0],
        sell_max: sellRange[1],

        sell_pct:
          sellPercentage / 100,

        fee_rate:
          feePercentage / 100,

        allocations:
          selectedAssets.map(
            (asset) => ({
              asset,

              weight:
                (
                  weights[asset] ?? 0
                ) / 100,
            }),
          ),
      };

    try {
      setRunning(true);
      setError("");
      setResult(null);

      const response =
        await createPortfolioBacktest(
          request,
          controller.signal,
        );

      setResult(response);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "投資組合回測執行失敗。",
      );
    } finally {
      setRunning(false);
    }
  }


  return (
    <section className="portfolio-backtest-page">
      <header className="portfolio-backtest-heading">
        <div>
          <h2>Portfolio Backtest</h2>

          <p>
            使用 BTC 風險指數控制買入與止盈，
            並依照設定權重分配多種加密資產。
          </p>
        </div>

        <div className="portfolio-backtest-period">
          <span>回測期間</span>

          <strong>
            {startDate}
            {" → "}
            {endDate}
          </strong>
        </div>
      </header>


      <form
        className="portfolio-backtest-form"
        onSubmit={handleSubmit}
      >
        <section className="portfolio-form-card">
          <div className="portfolio-card-heading">
            <div>
              <span className="portfolio-step">
                Step 1
              </span>

              <h3>基本參數</h3>
            </div>
          </div>

          <div className="portfolio-form-grid">
            <label>
              <span>每日投入預算（USDT）</span>

              <input
                type="number"
                min="1"
                step="1"
                value={dailyBudget}
                onChange={(event) =>
                  setDailyBudget(
                    Number(
                      event.target.value,
                    ),
                  )
                }
              />

              <small>
                只有符合買入風險時才會投入
              </small>
            </label>

            <label>
              <span>開始日期</span>

              <input
                type="date"
                min={HISTORY_START_DATE}
                max={endDate}
                value={startDate}
                onChange={(event) =>
                  setStartDate(
                    event.target.value,
                  )
                }
              />
            </label>

            <label>
              <span>結束日期</span>

              <input
                type="date"
                min={startDate}
                max={today}
                value={endDate}
                onChange={(event) =>
                  setEndDate(
                    event.target.value,
                  )
                }
              />
            </label>

            <label>
              <span>每次止盈比例（%）</span>

              <input
                type="number"
                min="0.01"
                max="100"
                step="0.01"
                value={sellPercentage}
                onChange={(event) =>
                  setSellPercentage(
                    Number(
                      event.target.value,
                    ),
                  )
                }
              />

              <small>
                每次賣出目前持倉的
                {" "}
                {sellPercentage.toFixed(2)}
                %
              </small>
            </label>

            <label>
              <span>交易手續費率（%）</span>

              <input
                type="number"
                min="0"
                max="10"
                step="0.01"
                value={feePercentage}
                onChange={(event) =>
                  setFeePercentage(
                    Number(
                      event.target.value,
                    ),
                  )
                }
              />

              <small>
                買入與賣出都會計算手續費
              </small>
            </label>
          </div>
        </section>


        <section className="portfolio-form-card">
          <div className="portfolio-card-heading">
            <div>
              <span className="portfolio-step">
                Step 2
              </span>

              <h3>風險區間</h3>
            </div>
          </div>

          <div className="risk-slider-grid">
            <RiskRangeSlider
              label="買入風險區間"
              description={
                "BTC 風險進入此區間時，" +
                "按權重買入所有幣種。"
              }
              value={buyRange}
              onChange={(value) => {
                setBuyRange(value);
                setResult(null);
              }}
              disabled={running}
            />

            <RiskRangeSlider
              label="止盈風險區間"
              description={
                "BTC 風險進入此區間時，" +
                "按設定比例賣出持倉。"
              }
              value={sellRange}
              onChange={(value) => {
                setSellRange(value);
                setResult(null);
              }}
              disabled={running}
            />
          </div>
        </section>


        <section className="portfolio-form-card">
          <div className="portfolio-card-heading">
            <div>
              <span className="portfolio-step">
                Step 3
              </span>

              <h3>選擇幣種</h3>
            </div>

            <span className="selected-count">
              已選擇
              {" "}
              {selectedAssets.length}
              {" "}
              種
            </span>
          </div>

          <div className="asset-selector">
            {SUPPORTED_ASSETS.map(
              (asset) => {
                const selected =
                  selectedAssets.includes(
                    asset,
                  );

                return (
                  <button
                    key={asset}
                    type="button"
                    className={
                      selected
                        ? "asset-chip selected"
                        : "asset-chip"
                    }
                    aria-pressed={selected}
                    onClick={() =>
                      handleAssetToggle(asset)
                    }
                    disabled={running}
                  >
                    <span
                      className="asset-chip-symbol"
                    >
                      {asset.slice(0, 1)}
                    </span>

                    {asset}

                    {selected && (
                      <span
                        className={
                          "asset-chip-check"
                        }
                      >
                        ✓
                      </span>
                    )}
                  </button>
                );
              },
            )}
          </div>
        </section>


        <section className="portfolio-form-card">
          <div className="portfolio-card-heading">
            <div>
              <span className="portfolio-step">
                Step 4
              </span>

              <h3>幣種權重</h3>
            </div>

            <strong
              className={
                Math.abs(
                  weightTotal - 100,
                ) <= 0.01
                  ? "weight-total valid"
                  : "weight-total invalid"
              }
            >
              總和
              {" "}
              {weightTotal.toFixed(2)}
              %
            </strong>
          </div>

          <div className="weight-allocation-list">
            {selectedAssets.map(
              (asset) => (
                <div
                  key={asset}
                  className={
                    "weight-allocation-row"
                  }
                >
                  <div
                    className={
                      "weight-asset-name"
                    }
                  >
                    <span
                      className={
                        "weight-asset-symbol"
                      }
                    >
                      {asset.slice(0, 1)}
                    </span>

                    <strong>{asset}</strong>
                  </div>

                  <div
                    className={
                      "weight-input-group"
                    }
                  >
                    <input
                      type="number"
                      min="0.01"
                      max="100"
                      step="0.01"
                      value={
                        weights[asset] ?? 0
                      }
                      onChange={(event) =>
                        updateWeight(
                          asset,
                          Number(
                            event.target.value,
                          ),
                        )
                      }
                      disabled={running}
                    />

                    <span>%</span>
                  </div>

                  <div
                    className={
                      "weight-progress-track"
                    }
                  >
                    <span
                      style={{
                        width:
                          `${
                            Math.min(
                              100,
                              Math.max(
                                0,
                                weights[asset] ??
                                  0,
                              ),
                            )
                          }%`,
                      }}
                    />
                  </div>
                </div>
              ),
            )}
          </div>

          <button
            type="button"
            className="equal-weight-button"
            onClick={() =>
              setWeights(
                createEqualWeights(
                  selectedAssets,
                ),
              )
            }
            disabled={running}
          >
            平均分配權重
          </button>
        </section>


        {error && (
          <div className="portfolio-backtest-error">
            <strong>無法執行回測</strong>
            <p>{error}</p>
          </div>
        )}


        <button
          type="submit"
          className="run-portfolio-button"
          disabled={running}
        >
          {running
            ? "正在取得資料並執行回測……"
            : "執行投資組合回測"}
        </button>
      </form>


      {result && (
        <section className="portfolio-result-section">
          <div className="portfolio-result-heading">
            <div>
              <span className="portfolio-step">
                Backtest Result
              </span>

              <h3>回測結果</h3>
            </div>

            <span>
              共
              {" "}
              {result.summary.trade_count}
              {" "}
              筆交易
            </span>
          </div>


          <div className="portfolio-summary-grid">
            <article
              className={
                result.summary.total_profit >= 0
                  ? "portfolio-summary-card positive"
                  : "portfolio-summary-card negative"
              }
            >
              <span>總獲利</span>

              <strong>
                {formatUsd(
                  result.summary.total_profit,
                )}
              </strong>

              <small>
                {formatMetricPercent(
                  result.summary.roi_pct,
                )}
                {" "}
                ROI
              </small>
            </article>

            <article className="portfolio-summary-card">
              <span>總權益</span>

              <strong>
                {formatUsd(
                  result.summary.total_equity,
                )}
              </strong>

              <small>
                現金加目前持倉市值
              </small>
            </article>

            <article
              className={
                "portfolio-summary-card negative"
              }
            >
              <span>最大回撤</span>

              <strong>
                {formatMetricPercent(
                  result.summary.mdd_pct,
                )}
              </strong>

              <small>
                Maximum Drawdown
              </small>
            </article>

            <article className="portfolio-summary-card">
              <span>累積投入</span>

              <strong>
                {formatUsd(
                  result.summary
                    .total_contributed,
                )}
              </strong>

              <small>
                觸發買入條件的投入總額
              </small>
            </article>

            <article className="portfolio-summary-card">
              <span>買入天數</span>

              <strong>
                {result.summary.buy_days}
              </strong>

              <small>天</small>
            </article>

            <article className="portfolio-summary-card">
              <span>止盈天數</span>

              <strong>
                {result.summary.sell_days}
              </strong>

              <small>天</small>
            </article>

            <article className="portfolio-summary-card">
              <span>現金餘額</span>

              <strong>
                {formatUsd(
                  result.summary.cash_balance,
                )}
              </strong>

              <small>已賣出後保留的現金</small>
            </article>

            <article className="portfolio-summary-card">
              <span>總手續費</span>

              <strong>
                {formatUsd(
                  result.summary.total_fees,
                )}
              </strong>

              <small>
                買入與賣出費用總和
              </small>
            </article>
          </div>
                
          <div className="portfolio-chart-grid">
            <PortfolioEquityChart
                points={result.equity_curve}
            />

            <AllocationChart
                assets={result.assets}
                cashBalance={
                result.summary.cash_balance
                }
            />
          </div>

          <section className="asset-result-card">
            <div className="asset-result-heading">
              <h3>各幣種績效</h3>

              <span>
                {result.assets.length}
                {" "}
                種資產
              </span>
            </div>

            <div className="asset-result-table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>幣種</th>
                    <th>目標權重</th>
                    <th>投入金額</th>
                    <th>目前市值</th>
                    <th>總損益</th>
                    <th>ROI</th>
                    <th>持有數量</th>
                    <th>手續費</th>
                  </tr>
                </thead>

                <tbody>
                  {result.assets.map(
                    (asset) => (
                      <tr key={asset.asset}>
                        <td>
                          <strong>
                            {asset.asset}
                          </strong>
                        </td>

                        <td>
                          {formatWeight(
                            asset.weight,
                          )}
                        </td>

                        <td>
                          {formatUsd(
                            asset.contributed,
                          )}
                        </td>

                        <td>
                          {formatUsd(
                            asset.market_value,
                          )}
                        </td>

                        <td
                          className={
                            asset.profit >= 0
                              ? "positive-text"
                              : "negative-text"
                          }
                        >
                          {formatUsd(
                            asset.profit,
                          )}
                        </td>

                        <td
                          className={
                            asset.roi_pct >= 0
                              ? "positive-text"
                              : "negative-text"
                          }
                        >
                          {formatMetricPercent(
                            asset.roi_pct,
                          )}
                        </td>

                        <td>
                          {asset.balance.toFixed(
                            8,
                          )}
                        </td>

                        <td>
                          {formatUsd(
                            asset.fees,
                          )}
                        </td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
          </section>
          <PortfolioTradeTable
            trades={result.trades}
          />
        </section>
      )}
    </section>
    
  );
}


export default BacktestPage;