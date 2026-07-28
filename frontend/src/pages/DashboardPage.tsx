import { useEffect, useState } from "react";

import {
  getLatestRisk,
  getRiskHistory,
} from "../api/risk";

import type {
  RiskHistoryResponse,
  StoredRiskPoint,
} from "../types/risk";

import "./DashboardPage.css";
import RiskTrendChart from "../components/charts/RiskTrendChart";

function formatPrice(price: number | null): string {
  if (price === null) {
    return "--";
  }

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(price);
}


function formatRisk(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}


function formatDate(value: string): string {
  return new Date(value).toLocaleString("zh-TW");
}


function DashboardPage() {
  const [latest, setLatest] =
    useState<StoredRiskPoint | null>(null);

  const [history, setHistory] =
    useState<RiskHistoryResponse | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");


  useEffect(() => {
    const controller = new AbortController();

    async function loadDashboard() {
      try {
        setLoading(true);
        setError("");

        const [
          latestResult,
          historyResult,
        ] = await Promise.all([
          getLatestRisk(
            "BTCUSDT",
            controller.signal,
          ),
          getRiskHistory(
            "BTCUSDT",
            30,
            controller.signal,
          ),
        ]);

        if (!controller.signal.aborted) {
          setLatest(latestResult);
          setHistory(historyResult);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          const message =
            err instanceof Error
              ? err.message
              : "無法載入 Dashboard 資料";

          setError(message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadDashboard();

    return () => {
      controller.abort();
    };
  }, []);


  if (loading) {
    return (
      <section className="dashboard-page">
        <h2>Dashboard</h2>
        <p>正在載入 BTC 風險資料……</p>
      </section>
    );
  }


  if (error) {
    return (
      <section className="dashboard-page">
        <h2>Dashboard</h2>

        <div className="error-message">
          <strong>資料載入失敗</strong>
          <p>{error}</p>
        </div>
      </section>
    );
  }


  if (!latest || !history) {
    return (
      <section className="dashboard-page">
        <h2>Dashboard</h2>
        <p>目前沒有可顯示的風險資料。</p>
      </section>
    );
  }


  const recentRecords = [
    ...history.records,
  ]
    .reverse()
    .slice(0, 10);


  return (
    <section className="dashboard-page">
      <div className="dashboard-heading">
        <div>
          <h2>BTC Risk Dashboard</h2>
          <p>
            最新資料時間：
            {formatDate(latest.score_time)}
          </p>
        </div>

        <span className="risk-level">
          {latest.risk_level}
        </span>
      </div>


      <div className="summary-grid">
        <article className="metric-card">
          <span>BTC Price</span>
          <strong>
            {formatPrice(latest.price)}
          </strong>
        </article>

        <article className="metric-card">
          <span>Total Risk</span>
          <strong>
            {formatRisk(latest.total_risk)}
          </strong>
        </article>

        <article className="metric-card">
          <span>Price Risk</span>
          <strong>
            {formatRisk(latest.price_risk)}
          </strong>
        </article>

        <article className="metric-card">
          <span>Social Risk</span>
          <strong>
            {formatRisk(latest.social_risk)}
          </strong>
        </article>
      </div>

      <RiskTrendChart
        records={history.records}
      />


      <section className="history-section">
        <div className="section-heading">
          <h3>Recent Risk Records</h3>
          <span>
            共取得 {history.count} 筆
          </span>
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>時間</th>
                <th>價格</th>
                <th>總風險</th>
                <th>價格風險</th>
                <th>社群風險</th>
                <th>等級</th>
              </tr>
            </thead>

            <tbody>
              {recentRecords.map((record) => (
                <tr key={record.score_time}>
                  <td>
                    {formatDate(record.score_time)}
                  </td>

                  <td>
                    {formatPrice(record.price)}
                  </td>

                  <td>
                    {formatRisk(record.total_risk)}
                  </td>

                  <td>
                    {formatRisk(record.price_risk)}
                  </td>

                  <td>
                    {formatRisk(record.social_risk)}
                  </td>

                  <td>
                    {record.risk_level}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}


export default DashboardPage;