import {
  useEffect,
  useState,
} from "react";

import {
  getDataStatus,
} from "../api/status";

import type {
  DataStatusResponse,
  SystemStatus,
} from "../types/status";

import "./DataStatusPage.css";


function formatDateTime(
  value: string | null,
): string {
  if (value === null) {
    return "--";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString(
    "zh-TW",
    {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    },
  );
}


function formatAge(
  value: number | null,
): string {
  if (value === null) {
    return "--";
  }

  if (value < 1) {
    return `${Math.round(
      value * 60,
    )} 分鐘`;
  }

  if (value < 24) {
    return `${value.toFixed(1)} 小時`;
  }

  return `${(
    value / 24
  ).toFixed(1)} 天`;
}


function getStatusLabel(
  status: SystemStatus,
): string {
  switch (status) {
    case "healthy":
      return "系統正常";

    case "degraded":
      return "部分異常";

    case "unavailable":
      return "服務無法使用";
  }
}


function getBooleanStatusLabel(
  value: boolean,
  positiveText: string,
  negativeText: string,
): string {
  return value
    ? positiveText
    : negativeText;
}


function DataStatusPage() {
  const [data, setData] =
    useState<DataStatusResponse | null>(
      null,
    );

  const [loading, setLoading] =
    useState(true);

  const [refreshing, setRefreshing] =
    useState(false);

  const [error, setError] =
    useState("");


  async function loadDataStatus(
    signal?: AbortSignal,
    isRefresh = false,
  ) {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError("");

      const response =
        await getDataStatus(signal);

      if (!signal?.aborted) {
        setData(response);
      }
    } catch (err) {
      if (!signal?.aborted) {
        setError(
          err instanceof Error
            ? err.message
            : "無法取得系統資料狀態。",
        );
      }
    } finally {
      if (!signal?.aborted) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }


  useEffect(() => {
    const controller =
      new AbortController();

    void loadDataStatus(
      controller.signal,
    );

    return () => {
      controller.abort();
    };
  }, []);


  function handleRefresh() {
    void loadDataStatus(
      undefined,
      true,
    );
  }


  return (
    <section className="data-status-page">
      <header className="data-status-heading">
        <div>
          <h2>Data Status</h2>

          <p>
            檢查後端服務、資料庫、
            BTC 風險資料與 Binance API
            的目前狀態。
          </p>
        </div>

        <button
          type="button"
          className="data-status-refresh-button"
          onClick={handleRefresh}
          disabled={
            loading ||
            refreshing
          }
        >
          {refreshing
            ? "正在重新檢查……"
            : "重新檢查"}
        </button>
      </header>


      {loading && (
        <div className="data-status-message">
          正在檢查系統狀態……
        </div>
      )}


      {error && (
        <div className="data-status-error">
          <div>
            <strong>
              無法取得 Data Status
            </strong>

            <p>{error}</p>
          </div>

          <button
            type="button"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            重試
          </button>
        </div>
      )}


      {!loading && data && (
        <>
          <section
            className={
              `overall-status-card ` +
              `status-${data.status}`
            }
          >
            <div>
              <span>Overall Status</span>

              <h3>
                {getStatusLabel(
                  data.status,
                )}
              </h3>

              <p>
                {data.status === "healthy" &&
                  "資料庫、風險資料與市場 API 皆正常。"}

                {data.status === "degraded" &&
                  "部分資料或外部服務出現異常，主要功能可能仍可使用。"}

                {data.status === "unavailable" &&
                  "資料庫目前無法使用，部分系統功能將無法正常運作。"}
              </p>
            </div>

            <div
              className={
                `status-indicator ` +
                `status-${data.status}`
              }
            >
              <span />
              {data.status.toUpperCase()}
            </div>
          </section>


          <section className="service-status-grid">
            <article className="service-status-card">
              <div className="service-status-card__heading">
                <div>
                  <span>Database</span>
                  <h3>資料庫</h3>
                </div>

                <span
                  className={
                    data.database.connected
                      ? "service-badge success"
                      : "service-badge danger"
                  }
                >
                  {getBooleanStatusLabel(
                    data.database.connected,
                    "Connected",
                    "Unavailable",
                  )}
                </span>
              </div>

              <div className="service-status-detail">
                <span>連線狀態</span>

                <strong>
                  {getBooleanStatusLabel(
                    data.database.connected,
                    "正常連線",
                    "無法連線",
                  )}
                </strong>
              </div>
            </article>


            <article className="service-status-card">
              <div className="service-status-card__heading">
                <div>
                  <span>Market API</span>
                  <h3>
                    {data.market_api.provider}
                  </h3>
                </div>

                <span
                  className={
                    data.market_api.available
                      ? "service-badge success"
                      : "service-badge warning"
                  }
                >
                  {getBooleanStatusLabel(
                    data.market_api.available,
                    "Available",
                    "Unavailable",
                  )}
                </span>
              </div>

              <div className="service-status-detail">
                <span>市場資料服務</span>

                <strong>
                  {getBooleanStatusLabel(
                    data.market_api.available,
                    "可正常存取",
                    "目前無法存取",
                  )}
                </strong>
              </div>
            </article>


            <article className="service-status-card">
              <div className="service-status-card__heading">
                <div>
                  <span>Risk Data</span>
                  <h3>
                    {data.risk_data.symbol}
                  </h3>
                </div>

                <span
                  className={
                    data.risk_data.is_stale
                      ? "service-badge warning"
                      : "service-badge success"
                  }
                >
                  {data.risk_data.is_stale
                    ? "Stale"
                    : "Fresh"}
                </span>
              </div>

              <div className="service-status-detail">
                <span>資料新鮮度</span>

                <strong>
                  {data.risk_data.is_stale
                    ? "資料可能已過期"
                    : "資料仍在有效期限內"}
                </strong>
              </div>
            </article>
          </section>


          <section className="risk-data-card">
            <div className="risk-data-card__heading">
              <div>
                <span>Risk Dataset</span>

                <h3>
                  BTC 風險歷史資料
                </h3>
              </div>

              <strong>
                {data.risk_data
                  .record_count
                  .toLocaleString()}
                {" "}
                records
              </strong>
            </div>


            <div className="risk-data-metric-grid">
              <article>
                <span>資料筆數</span>

                <strong>
                  {data.risk_data
                    .record_count
                    .toLocaleString()}
                </strong>

                <small>
                  Valid risk records
                </small>
              </article>

              <article>
                <span>最早資料時間</span>

                <strong>
                  {formatDateTime(
                    data.risk_data
                      .earliest_time,
                  )}
                </strong>

                <small>
                  Earliest record
                </small>
              </article>

              <article>
                <span>最新資料時間</span>

                <strong>
                  {formatDateTime(
                    data.risk_data
                      .latest_time,
                  )}
                </strong>

                <small>
                  Latest record
                </small>
              </article>

              <article>
                <span>距離最新資料</span>

                <strong>
                  {formatAge(
                    data.risk_data
                      .age_hours,
                  )}
                </strong>

                <small>
                  Data age
                </small>
              </article>
            </div>
          </section>


          <footer className="data-status-footer">
            <span>
              最後檢查時間
            </span>

            <strong>
              {formatDateTime(
                data.checked_at,
              )}
            </strong>
          </footer>
        </>
      )}
    </section>
  );
}


export default DataStatusPage;