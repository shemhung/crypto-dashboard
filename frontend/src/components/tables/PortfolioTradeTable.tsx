import {
  useMemo,
  useState,
} from "react";

import type {
  PortfolioTrade,
} from "../../types/backtest";

import "./PortfolioTradeTable.css";


interface PortfolioTradeTableProps {
  trades: PortfolioTrade[];
}


type TradeTypeFilter =
  | "ALL"
  | "BUY"
  | "SELL";


const PAGE_SIZE = 50;


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


function formatDate(
  value: string,
): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString(
    "zh-TW",
    {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    },
  );
}


function formatRisk(
  value: number,
): string {
  return [
    value.toFixed(3),
    `(${(value * 100).toFixed(1)}%)`,
  ].join(" ");
}


function PortfolioTradeTable({
  trades,
}: PortfolioTradeTableProps) {
  const [
    assetFilter,
    setAssetFilter,
  ] = useState("ALL");

  const [
    tradeTypeFilter,
    setTradeTypeFilter,
  ] = useState<TradeTypeFilter>(
    "ALL",
  );

  const [page, setPage] =
    useState(1);


  const availableAssets = useMemo(
    () => {
      return Array.from(
        new Set(
          trades.map(
            (trade) => trade.asset,
          ),
        ),
      ).sort();
    },
    [trades],
  );


  const filteredTrades = useMemo(
    () => {
      return [...trades]
        .filter((trade) => {
          const matchesAsset =
            assetFilter === "ALL" ||
            trade.asset === assetFilter;

          const matchesType =
            tradeTypeFilter === "ALL" ||
            trade.type === tradeTypeFilter;

          return (
            matchesAsset &&
            matchesType
          );
        })
        .sort((first, second) => {
          return (
            new Date(
              second.date,
            ).getTime() -
            new Date(
              first.date,
            ).getTime()
          );
        });
    },
    [
      trades,
      assetFilter,
      tradeTypeFilter,
    ],
  );


  const totalPages = Math.max(
    1,
    Math.ceil(
      filteredTrades.length /
      PAGE_SIZE,
    ),
  );

  const currentPage = Math.min(
    page,
    totalPages,
  );

  const pageStart =
    (currentPage - 1) * PAGE_SIZE;

  const visibleTrades =
    filteredTrades.slice(
      pageStart,
      pageStart + PAGE_SIZE,
    );


  const tradeSummary = useMemo(
    () => {
      return filteredTrades.reduce(
        (
          summary,
          trade,
        ) => {
          if (trade.type === "BUY") {
            summary.buyValue +=
              trade.value_usdt;

            summary.buyCount += 1;
          } else {
            summary.sellValue +=
              trade.value_usdt;

            summary.sellCount += 1;
          }

          summary.totalFees +=
            trade.fee;

          return summary;
        },
        {
          buyValue: 0,
          sellValue: 0,
          totalFees: 0,
          buyCount: 0,
          sellCount: 0,
        },
      );
    },
    [filteredTrades],
  );


  function handleAssetChange(
    value: string,
  ) {
    setAssetFilter(value);
    setPage(1);
  }


  function handleTypeChange(
    value: TradeTypeFilter,
  ) {
    setTradeTypeFilter(value);
    setPage(1);
  }


  if (trades.length === 0) {
    return (
      <section className="portfolio-trade-card">
        <div className="portfolio-trade-heading">
          <div>
            <span>Trade History</span>
            <h3>交易紀錄</h3>
          </div>
        </div>

        <p className="portfolio-trade-empty">
          此次回測沒有產生任何交易。
        </p>
      </section>
    );
  }


  return (
    <section className="portfolio-trade-card">
      <div className="portfolio-trade-heading">
        <div>
          <span>Trade History</span>
          <h3>交易紀錄</h3>
        </div>

        <strong>
          {filteredTrades.length.toLocaleString()}
          {" "}
          筆
        </strong>
      </div>


      <div className="portfolio-trade-controls">
        <label>
          <span>幣種</span>

          <select
            value={assetFilter}
            onChange={(event) =>
              handleAssetChange(
                event.target.value,
              )
            }
          >
            <option value="ALL">
              全部幣種
            </option>

            {availableAssets.map(
              (asset) => (
                <option
                  key={asset}
                  value={asset}
                >
                  {asset}
                </option>
              ),
            )}
          </select>
        </label>


        <div className="trade-type-filter">
          <span>交易類型</span>

          <div>
            <button
              type="button"
              className={
                tradeTypeFilter === "ALL"
                  ? "active"
                  : ""
              }
              onClick={() =>
                handleTypeChange("ALL")
              }
            >
              全部
            </button>

            <button
              type="button"
              className={
                tradeTypeFilter === "BUY"
                  ? "active"
                  : ""
              }
              onClick={() =>
                handleTypeChange("BUY")
              }
            >
              BUY
            </button>

            <button
              type="button"
              className={
                tradeTypeFilter === "SELL"
                  ? "active"
                  : ""
              }
              onClick={() =>
                handleTypeChange("SELL")
              }
            >
              SELL
            </button>
          </div>
        </div>
      </div>


      <div className="portfolio-trade-summary">
        <article>
          <span>買入筆數</span>

          <strong>
            {tradeSummary.buyCount}
          </strong>
        </article>

        <article>
          <span>買入總額</span>

          <strong>
            {formatUsd(
              tradeSummary.buyValue,
            )}
          </strong>
        </article>

        <article>
          <span>賣出筆數</span>

          <strong>
            {tradeSummary.sellCount}
          </strong>
        </article>

        <article>
          <span>賣出總額</span>

          <strong>
            {formatUsd(
              tradeSummary.sellValue,
            )}
          </strong>
        </article>

        <article>
          <span>交易手續費</span>

          <strong>
            {formatUsd(
              tradeSummary.totalFees,
            )}
          </strong>
        </article>
      </div>


      {filteredTrades.length === 0 ? (
        <p className="portfolio-trade-empty">
          目前篩選條件沒有符合的交易。
        </p>
      ) : (
        <>
          <div className="portfolio-trade-table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>日期</th>
                  <th>幣種</th>
                  <th>類型</th>
                  <th>價格</th>
                  <th>風險值</th>
                  <th>交易數量</th>
                  <th>交易金額</th>
                  <th>手續費</th>
                </tr>
              </thead>

              <tbody>
                {visibleTrades.map(
                  (trade, index) => (
                    <tr
                      key={
                        `${trade.date}-` +
                        `${trade.asset}-` +
                        `${trade.type}-` +
                        `${pageStart + index}`
                      }
                    >
                      <td>
                        {formatDate(
                          trade.date,
                        )}
                      </td>

                      <td>
                        <strong>
                          {trade.asset}
                        </strong>
                      </td>

                      <td>
                        <span
                          className={
                            trade.type ===
                            "BUY"
                              ? "portfolio-trade-type buy"
                              : "portfolio-trade-type sell"
                          }
                        >
                          {trade.type}
                        </span>
                      </td>

                      <td>
                        {formatUsd(
                          trade.price,
                        )}
                      </td>

                      <td>
                        {formatRisk(
                          trade.risk,
                        )}
                      </td>

                      <td>
                        {trade.amount.toFixed(
                          8,
                        )}
                      </td>

                      <td>
                        {formatUsd(
                          trade.value_usdt,
                        )}
                      </td>

                      <td>
                        {formatUsd(
                          trade.fee,
                        )}
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>


          <div className="portfolio-trade-pagination">
            <span>
              第
              {" "}
              {currentPage}
              {" "}
              頁，共
              {" "}
              {totalPages}
              {" "}
              頁
            </span>

            <div>
              <button
                type="button"
                disabled={
                  currentPage <= 1
                }
                onClick={() =>
                  setPage(
                    currentPage - 1,
                  )
                }
              >
                上一頁
              </button>

              <button
                type="button"
                disabled={
                  currentPage >=
                  totalPages
                }
                onClick={() =>
                  setPage(
                    currentPage + 1,
                  )
                }
              >
                下一頁
              </button>
            </div>
          </div>
        </>
      )}
    </section>
  );
}


export default PortfolioTradeTable;