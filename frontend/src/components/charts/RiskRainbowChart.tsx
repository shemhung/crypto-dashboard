import type {
  CustomSeriesRenderItemAPI,
  CustomSeriesRenderItemParams,
  CustomSeriesRenderItemReturn,
  EChartsOption,
} from "echarts";

import ReactECharts from "echarts-for-react";

import type { StoredRiskPoint } from "../../types/risk";

import "./RiskRainbowChart.css";


interface RiskRainbowChartProps {
  records: StoredRiskPoint[];
}


/**
 * 普通折線使用的資料：
 * [時間, 價格, 風險]
 */
type ChartPoint = [
  timestamp: number,
  price: number,
  risk: number,
];


/**
 * Custom Series 使用的線段資料：
 * [
 *   起點時間,
 *   起點價格,
 *   終點時間,
 *   終點價格,
 *   該線段風險
 * ]
 */
type RiskSegment = [
  startTimestamp: number,
  startPrice: number,
  endTimestamp: number,
  endPrice: number,
  risk: number,
];


interface TooltipParameter {
  seriesName?: string;
  data?: unknown;
}


const riskPieces = [
  {
    gte: 0.0,
    lt: 0.1,
    label: "0.0 - 0.1",
    color: "#60a5fa",
  },
  {
    gte: 0.1,
    lt: 0.2,
    label: "0.1 - 0.2",
    color: "#7dd3fc",
  },
  {
    gte: 0.2,
    lt: 0.3,
    label: "0.2 - 0.3",
    color: "#93c5fd",
  },
  {
    gte: 0.3,
    lt: 0.4,
    label: "0.3 - 0.4",
    color: "#a7f3d0",
  },
  {
    gte: 0.4,
    lt: 0.5,
    label: "0.4 - 0.5",
    color: "#86efac",
  },
  {
    gte: 0.5,
    lt: 0.6,
    label: "0.5 - 0.6",
    color: "#fde68a",
  },
  {
    gte: 0.6,
    lt: 0.7,
    label: "0.6 - 0.7",
    color: "#fcd34d",
  },
  {
    gte: 0.7,
    lt: 0.8,
    label: "0.7 - 0.8",
    color: "#fdba74",
  },
  {
    gte: 0.8,
    lt: 0.85,
    label: "0.8 - 0.85",
    color: "#fb923c",
  },
  {
    gte: 0.85,
    lte: 1.0,
    label: "0.85 - 1.0",
    color: "#f87171",
  },
];


function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}


function formatPriceAxis(value: number): string {
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`;
  }

  if (value >= 1000) {
    return `$${Math.round(value / 1000)}K`;
  }

  return `$${value}`;
}


function isTooltipParameter(
  value: unknown,
): value is TooltipParameter {
  return (
    typeof value === "object" &&
    value !== null
  );
}


function isChartPoint(
  value: unknown,
): value is ChartPoint {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every(
      (item) => typeof item === "number",
    )
  );
}


function formatTooltip(params: unknown): string {
  const parameterList = (
    Array.isArray(params)
      ? params
      : [params]
  ).filter(isTooltipParameter);

  /*
   * 找透明折線的資料。
   * 不使用 custom series 的五欄線段資料。
   */
  const tooltipParameter =
    parameterList.find(
      (parameter) =>
        parameter.seriesName ===
        "BTC Tooltip Data",
    ) ??
    parameterList.find(
      (parameter) =>
        isChartPoint(parameter.data),
    );

  if (
    tooltipParameter === undefined ||
    !isChartPoint(tooltipParameter.data)
  ) {
    return "";
  }

  const [
    timestamp,
    price,
    risk,
  ] = tooltipParameter.data;

  const date = new Date(
    timestamp,
  ).toLocaleDateString("zh-TW", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });

  return [
    `<strong>${date}</strong>`,
    `BTC Price：${formatUsd(price)}`,
    `Total Risk：${risk.toFixed(3)}`,
    `Risk Percentage：${(risk * 100).toFixed(1)}%`,
  ].join("<br />");
}


/**
 * 每呼叫一次，就畫一小段 BTC 價格線。
 *
 * 線段顏色由 visualMap 根據第 5 個欄位 risk 決定。
 */
function renderRiskSegment(
  _params: CustomSeriesRenderItemParams,
  api: CustomSeriesRenderItemAPI,
): CustomSeriesRenderItemReturn {
  const startTimestamp = Number(api.value(0));
  const startPrice = Number(api.value(1));

  const endTimestamp = Number(api.value(2));
  const endPrice = Number(api.value(3));

  const startPoint = api.coord([
    startTimestamp,
    startPrice,
  ]);

  const endPoint = api.coord([
    endTimestamp,
    endPrice,
  ]);

  const segmentColor = String(
    api.visual("color"),
  );

  return {
    type: "line",

    shape: {
      x1: startPoint[0],
      y1: startPoint[1],
      x2: endPoint[0],
      y2: endPoint[1],
    },

    style: {
      stroke: segmentColor,
      lineWidth: 2.5,
      lineCap: "round",
      lineJoin: "round",
    },

    silent: true,
  };
}


function RiskRainbowChart({
  records,
}: RiskRainbowChartProps) {
  /*
   * 先建立普通歷史資料：
   * [timestamp, price, totalRisk]
   */
  const chartData: ChartPoint[] = records
    .filter((record) => {
      return (
        record.price !== null &&
        record.price > 0 &&
        Number.isFinite(record.price) &&
        Number.isFinite(record.total_risk)
      );
    })
    .map((record): ChartPoint => [
      new Date(record.score_time).getTime(),
      record.price as number,
      record.total_risk,
    ])
    .sort(
      (first, second) =>
        first[0] - second[0],
    );


  if (chartData.length === 0) {
    return (
      <section className="rainbow-chart-card">
        <h3>BTC Cycle Risk Map</h3>

        <p>
          目前沒有可顯示的歷史風險資料。
        </p>
      </section>
    );
  }


  /*
   * 將每兩個相鄰資料點轉成一條線段。
   *
   * 例如：
   * 第 1 天 → 第 2 天
   * 第 2 天 → 第 3 天
   * 第 3 天 → 第 4 天
   */
  const segmentData: RiskSegment[] =
    chartData
      .slice(1)
      .map((currentPoint, index) => {
        const previousPoint =
          chartData[index];

        return [
          previousPoint[0],
          previousPoint[1],
          currentPoint[0],
          currentPoint[1],

          // 使用線段終點當天的 risk 決定顏色
          currentPoint[2],
        ];
      });


  const option: EChartsOption = {
    animation: false,

    title: {
      text: "BTC Price Colored by Risk Level",
      subtext:
        "BTC 價格依照 Total Risk 區間分段著色",
      left: 20,
      top: 10,
    },

    grid: {
      left: 80,
      right: 190,
      top: 90,
      bottom: 100,
    },

    tooltip: {
      trigger: "axis",

      axisPointer: {
        type: "cross",
      },

      formatter: formatTooltip,
    },

    toolbox: {
      right: 20,

      feature: {
        dataZoom: {
          yAxisIndex: "none",
        },

        restore: {},

        saveAsImage: {
          name: "btc-cycle-risk-map",
        },
      },
    },

    xAxis: {
      type: "time",

      name: "Date",
      nameLocation: "middle",
      nameGap: 35,

      axisLabel: {
        formatter: (
          value: number | string,
        ) => {
          return String(
            new Date(
              Number(value),
            ).getFullYear(),
          );
        },
      },
    },

    yAxis: {
      type: "log",
      logBase: 10,

      name: "BTC Price (USD)",
      nameLocation: "middle",
      nameGap: 58,

      min: "dataMin",
      max: "dataMax",

      axisLabel: {
        formatter: (
          value: number | string,
        ) => {
          return formatPriceAxis(
            Number(value),
          );
        },
      },

      splitLine: {
        show: true,
      },
    },

    /*
     * segmentData 的第 5 個值是 risk，
     * 所以這裡使用 dimension: 4。
     *
     * ECharts 的陣列索引從 0 開始：
     * 0 起點時間
     * 1 起點價格
     * 2 終點時間
     * 3 終點價格
     * 4 risk
     */
    visualMap: {
      type: "piecewise",
      dimension: 4,
      seriesIndex: 0,

      right: 15,
      top: 95,
      orient: "vertical",

      itemWidth: 22,
      itemHeight: 12,

      text: [
        "Higher Risk",
        "Lower Risk",
      ],

      pieces: riskPieces,

      outOfRange: {
        color: "#9ca3af",
      },
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
        bottom: 30,
        height: 24,
      },
    ],

    series: [
      /*
       * 第一層：真正看得到的彩色線段。
       */
      {
        name: "BTC Price by Risk",

        type: "custom",
        coordinateSystem: "cartesian2d",

        dimensions: [
          "startTimestamp",
          "startPrice",
          "endTimestamp",
          "endPrice",
          "risk",
        ],

        encode: {
          x: [0, 2],
          y: [1, 3],
        },

        data: segmentData,
        renderItem: renderRiskSegment,

        clip: true,
        silent: true,
        z: 3,

        tooltip: {
          show: false,
        },
      },

      /*
       * 第二層：透明的普通折線。
       *
       * 使用者看不到它，但它負責：
       * 1. Tooltip
       * 2. 十字游標
       * 3. 找到最接近滑鼠的日期
       */
      {
        name: "BTC Tooltip Data",

        type: "line",

        dimensions: [
          "timestamp",
          "price",
          "risk",
        ],

        encode: {
          x: 0,
          y: 1,
          tooltip: [1, 2],
        },

        data: chartData,

        showSymbol: false,
        connectNulls: false,

        lineStyle: {
          opacity: 0,
          width: 8,
        },

        itemStyle: {
          opacity: 0,
        },

        emphasis: {
          disabled: true,
        },

        z: 10,
      },
    ],
  };


  return (
    <section className="rainbow-chart-card">
      <div className="rainbow-chart-summary">
        <span>
          {chartData.length.toLocaleString()}
          {" "}
          records
        </span>

        <span>
          Logarithmic price scale
        </span>
      </div>

      <ReactECharts
        option={option}
        notMerge
        lazyUpdate
        style={{
          width: "100%",
          height: "640px",
        }}
      />
    </section>
  );
}


export default RiskRainbowChart;
