import type {
  EChartsOption,
} from "echarts";

import ReactECharts from "echarts-for-react";

import type {
  PortfolioEquityPoint,
} from "../../types/backtest";

import "./PortfolioEquityChart.css";


interface PortfolioEquityChartProps {
  points: PortfolioEquityPoint[];
}


interface TooltipParameter {
  seriesName?: string;
  value?: unknown;
  marker?: string;
}


type ChartValue = [
  timestamp: number,
  amount: number,
];


function formatUsd(
  value: number,
): string {
  return new Intl.NumberFormat(
    "zh-TW",
    {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    },
  ).format(value);
}


function formatAxisAmount(
  value: number,
): string {
  if (Math.abs(value) >= 1_000_000) {
    return `$${(
      value / 1_000_000
    ).toFixed(1)}M`;
  }

  if (Math.abs(value) >= 1_000) {
    return `$${(
      value / 1_000
    ).toFixed(0)}K`;
  }

  return `$${value.toFixed(0)}`;
}


function isChartValue(
  value: unknown,
): value is ChartValue {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number"
  );
}


function formatTooltip(
  params: unknown,
): string {
  const parameterList =
    (
      Array.isArray(params)
        ? params
        : [params]
    ) as TooltipParameter[];

  const firstValue =
    parameterList.find(
      (parameter) =>
        isChartValue(parameter.value),
    );

  if (
    firstValue === undefined ||
    !isChartValue(firstValue.value)
  ) {
    return "";
  }

  const date = new Date(
    firstValue.value[0],
  ).toLocaleDateString(
    "zh-TW",
    {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    },
  );

  const rows = parameterList
    .filter(
      (
        parameter,
      ): parameter is TooltipParameter & {
        value: ChartValue;
      } => isChartValue(parameter.value),
    )
    .map((parameter) => {
      return [
        parameter.marker ?? "",
        parameter.seriesName ?? "",
        "：",
        formatUsd(parameter.value[1]),
      ].join("");
    });

  return [
    `<strong>${date}</strong>`,
    ...rows,
  ].join("<br />");
}


function PortfolioEquityChart({
  points,
}: PortfolioEquityChartProps) {
  const sortedPoints = [...points]
    .filter((point) => {
      return (
        Number.isFinite(point.equity) &&
        Number.isFinite(
          point.contributed,
        )
      );
    })
    .sort((first, second) => {
      return (
        new Date(first.date).getTime() -
        new Date(second.date).getTime()
      );
    });


  if (sortedPoints.length === 0) {
    return (
      <section className="portfolio-equity-chart-card">
        <h3>組合淨值走勢</h3>

        <p>目前沒有可顯示的淨值資料。</p>
      </section>
    );
  }


  const equityData: ChartValue[] =
    sortedPoints.map((point) => [
      new Date(point.date).getTime(),
      point.equity,
    ]);


  const contributedData: ChartValue[] =
    sortedPoints.map((point) => [
      new Date(point.date).getTime(),
      point.contributed,
    ]);


  const option: EChartsOption = {
    animation: false,

    title: {
      text: "Portfolio Equity",
      subtext:
        "組合總權益與累積投入本金",
      left: 18,
      top: 12,
    },

    legend: {
      top: 18,
      right: 20,
      data: [
        "組合總權益",
        "累積投入本金",
      ],
    },

    grid: {
      left: 82,
      right: 30,
      top: 90,
      bottom: 90,
    },

    tooltip: {
      trigger: "axis",

      axisPointer: {
        type: "cross",
      },

      formatter: formatTooltip,
    },

    toolbox: {
      right: 18,
      top: 48,

      feature: {
        dataZoom: {
          yAxisIndex: "none",
        },

        restore: {},

        saveAsImage: {
          name:
            "portfolio-equity-chart",
        },
      },
    },

    xAxis: {
      type: "time",

      name: "Date",
      nameLocation: "middle",
      nameGap: 35,
    },

    yAxis: {
      type: "value",

      name: "USDT",
      nameLocation: "middle",
      nameGap: 65,

      axisLabel: {
        formatter: (
          value: number | string,
        ) => {
          return formatAxisAmount(
            Number(value),
          );
        },
      },

      splitLine: {
        show: true,
      },

      scale: true,
    },

    dataZoom: [
      {
        type: "inside",
        xAxisIndex: 0,
        filterMode: "none",
      },

      {
        type: "slider",
        xAxisIndex: 0,
        filterMode: "none",
        bottom: 28,
        height: 24,
      },
    ],

    series: [
      {
        name: "組合總權益",
        type: "line",
        data: equityData,

        showSymbol: false,
        smooth: false,

        lineStyle: {
          width: 2.5,
        },

        areaStyle: {
          opacity: 0.12,
        },

        emphasis: {
          focus: "series",
        },
      },

      {
        name: "累積投入本金",
        type: "line",
        data: contributedData,

        showSymbol: false,
        smooth: false,

        lineStyle: {
          width: 2,
          type: "dashed",
        },

        emphasis: {
          focus: "series",
        },
      },
    ],
  };


  return (
    <section className="portfolio-equity-chart-card">
      <div className="portfolio-equity-chart-summary">
        <span>
          {sortedPoints.length.toLocaleString()}
          {" "}
          records
        </span>

        <span>
          Equity = Cash + Market Value
        </span>
      </div>

      <ReactECharts
        option={option}
        notMerge
        lazyUpdate
        style={{
          width: "100%",
          height: "520px",
        }}
      />
    </section>
  );
}


export default PortfolioEquityChart;