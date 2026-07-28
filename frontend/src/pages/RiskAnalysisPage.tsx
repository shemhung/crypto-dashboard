import {
  useEffect,
  useMemo,
  useState,
} from "react";

import { getRiskHistoryByDateRange } from "../api/risk";
import RiskRainbowChart from "../components/charts/RiskRainbowChart";

import type { StoredRiskPoint } from "../types/risk";

import "./RiskAnalysisPage.css";


const HISTORY_START_DATE = "2017-08-17";


function getCurrentLocalDate(): string {
  const now = new Date();

  const year = now.getFullYear();
  const month = String(
    now.getMonth() + 1,
  ).padStart(2, "0");

  const day = String(
    now.getDate(),
  ).padStart(2, "0");

  return `${year}-${month}-${day}`;
}


function RiskAnalysisPage() {
  const endDate = useMemo(
    () => getCurrentLocalDate(),
    [],
  );

  const [records, setRecords] =
    useState<StoredRiskPoint[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");


  useEffect(() => {
    const controller = new AbortController();

    async function loadHistory() {
      try {
        setLoading(true);
        setError("");

        const result =
          await getRiskHistoryByDateRange(
            "BTCUSDT",
            HISTORY_START_DATE,
            endDate,
            controller.signal,
          );

        if (!controller.signal.aborted) {
          setRecords(result.records);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "無法載入歷史風險資料",
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadHistory();

    return () => {
      controller.abort();
    };
  }, [endDate]);


  if (loading) {
    return (
      <section className="risk-analysis-page">
        <h2>Risk Analysis</h2>

        <p>
          正在載入 2017 年至今的歷史資料……
        </p>
      </section>
    );
  }


  if (error) {
    return (
      <section className="risk-analysis-page">
        <h2>Risk Analysis</h2>

        <div className="risk-analysis-error">
          <strong>歷史資料載入失敗</strong>
          <p>{error}</p>
        </div>
      </section>
    );
  }


  return (
    <section className="risk-analysis-page">
      <header className="risk-analysis-heading">
        <div>
          <h2>Risk Analysis</h2>

          <p>
            觀察 BTC 價格在不同風險區間中的
            歷史週期變化。
          </p>
        </div>

        <div className="date-range">
          <span>From</span>
          <strong>{HISTORY_START_DATE}</strong>

          <span>To</span>
          <strong>{endDate}</strong>
        </div>
      </header>

      <RiskRainbowChart
        records={records}
      />
    </section>
  );
}


export default RiskAnalysisPage;