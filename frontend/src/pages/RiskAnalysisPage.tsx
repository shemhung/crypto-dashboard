import {
  useEffect,
  useState,
} from "react";

import {
  getRiskHistoryByDateRange,
} from "../api/risk";

import RiskRainbowChart from "../components/charts/RiskRainbowChart";

import type {
  StoredRiskPoint,
} from "../types/risk";

import "./RiskAnalysisPage.css";


const HISTORY_START_DATE = "2017-08-17";


type TimeRange =
  | "1Y"
  | "3Y"
  | "5Y"
  | "ALL"
  | "CUSTOM";


function formatLocalDate(
  value: Date,
): string {
  const year = value.getFullYear();

  const month = String(
    value.getMonth() + 1,
  ).padStart(2, "0");

  const day = String(
    value.getDate(),
  ).padStart(2, "0");

  return `${year}-${month}-${day}`;
}


function getCurrentLocalDate(): string {
  return formatLocalDate(new Date());
}


function subtractYears(
  endDate: string,
  years: number,
): string {
  const date = new Date(
    `${endDate}T00:00:00`,
  );

  date.setFullYear(
    date.getFullYear() - years,
  );

  return formatLocalDate(date);
}


function getPresetStartDate(
  range: Exclude<
    TimeRange,
    "CUSTOM"
  >,
  endDate: string,
): string {
  switch (range) {
    case "1Y":
      return subtractYears(endDate, 1);

    case "3Y":
      return subtractYears(endDate, 3);

    case "5Y":
      return subtractYears(endDate, 5);

    case "ALL":
      return HISTORY_START_DATE;
  }
}


function RiskAnalysisPage() {
  const today = getCurrentLocalDate();

  const [selectedRange, setSelectedRange] =
    useState<TimeRange>("ALL");

  /*
   * 真正送給後端 API 的日期。
   */
  const [queryStartDate, setQueryStartDate] =
    useState(HISTORY_START_DATE);

  const [queryEndDate, setQueryEndDate] =
    useState(today);

  /*
   * 自訂日期輸入框的暫存值。
   * 按下「套用」後才送出查詢。
   */
  const [customStartDate, setCustomStartDate] =
    useState(HISTORY_START_DATE);

  const [customEndDate, setCustomEndDate] =
    useState(today);

  const [records, setRecords] =
    useState<StoredRiskPoint[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [dateError, setDateError] =
    useState("");


  useEffect(() => {
    const controller =
      new AbortController();

    async function loadHistory() {
      try {
        setLoading(true);
        setError("");

        const result =
          await getRiskHistoryByDateRange(
            "BTCUSDT",
            queryStartDate,
            queryEndDate,
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
  }, [
    queryStartDate,
    queryEndDate,
  ]);


  function selectPresetRange(
    range: Exclude<
      TimeRange,
      "CUSTOM"
    >,
  ) {
    const endDate =
      getCurrentLocalDate();

    const startDate =
      getPresetStartDate(
        range,
        endDate,
      );

    setSelectedRange(range);
    setDateError("");

    setQueryStartDate(startDate);
    setQueryEndDate(endDate);
  }


  function showCustomRange() {
    setSelectedRange("CUSTOM");
    setDateError("");
  }


  function applyCustomRange() {
    if (
      !customStartDate ||
      !customEndDate
    ) {
      setDateError(
        "請選擇開始日期與結束日期。",
      );

      return;
    }

    if (
      customStartDate <
      HISTORY_START_DATE
    ) {
      setDateError(
        `開始日期不可早於 ${HISTORY_START_DATE}。`,
      );

      return;
    }

    if (
      customStartDate >
      customEndDate
    ) {
      setDateError(
        "開始日期不可晚於結束日期。",
      );

      return;
    }

    if (customEndDate > today) {
      setDateError(
        "結束日期不可晚於今天。",
      );

      return;
    }

    setDateError("");

    setQueryStartDate(
      customStartDate,
    );

    setQueryEndDate(
      customEndDate,
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

        <div className="current-date-range">
          <span>目前範圍</span>

          <strong>
            {queryStartDate}
            {" → "}
            {queryEndDate}
          </strong>
        </div>
      </header>


      <section className="range-control-card">
        <div className="preset-buttons">
          <button
            type="button"
            className={
              selectedRange === "1Y"
                ? "active"
                : ""
            }
            onClick={() =>
              selectPresetRange("1Y")
            }
          >
            1Y
          </button>

          <button
            type="button"
            className={
              selectedRange === "3Y"
                ? "active"
                : ""
            }
            onClick={() =>
              selectPresetRange("3Y")
            }
          >
            3Y
          </button>

          <button
            type="button"
            className={
              selectedRange === "5Y"
                ? "active"
                : ""
            }
            onClick={() =>
              selectPresetRange("5Y")
            }
          >
            5Y
          </button>

          <button
            type="button"
            className={
              selectedRange === "ALL"
                ? "active"
                : ""
            }
            onClick={() =>
              selectPresetRange("ALL")
            }
          >
            ALL
          </button>

          <button
            type="button"
            className={
              selectedRange === "CUSTOM"
                ? "active"
                : ""
            }
            onClick={showCustomRange}
          >
            自訂日期
          </button>
        </div>


        {selectedRange === "CUSTOM" && (
          <div className="custom-range-controls">
            <label>
              <span>開始日期</span>

              <input
                type="date"
                min={HISTORY_START_DATE}
                max={customEndDate}
                value={customStartDate}
                onChange={(event) =>
                  setCustomStartDate(
                    event.target.value,
                  )
                }
              />
            </label>

            <label>
              <span>結束日期</span>

              <input
                type="date"
                min={customStartDate}
                max={today}
                value={customEndDate}
                onChange={(event) =>
                  setCustomEndDate(
                    event.target.value,
                  )
                }
              />
            </label>

            <button
              type="button"
              className="apply-range-button"
              onClick={applyCustomRange}
            >
              套用
            </button>
          </div>
        )}


        {dateError && (
          <p className="date-error">
            {dateError}
          </p>
        )}
      </section>


      {loading && (
        <div className="risk-analysis-message">
          正在載入
          {" "}
          {queryStartDate}
          {" "}
          至
          {" "}
          {queryEndDate}
          {" "}
          的歷史資料……
        </div>
      )}


      {error && (
        <div className="risk-analysis-error">
          <strong>
            歷史資料載入失敗
          </strong>

          <p>{error}</p>
        </div>
      )}


      {!loading && !error && (
        <>
          <div className="risk-record-summary">
            <span>資料筆數</span>

            <strong>
              {records.length.toLocaleString()}
            </strong>
          </div>

          <RiskRainbowChart
            records={records}
          />
        </>
      )}
    </section>
  );
}


export default RiskAnalysisPage;