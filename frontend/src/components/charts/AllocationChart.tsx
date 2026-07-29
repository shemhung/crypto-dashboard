import type {
  EChartsOption,
} from "echarts";

import ReactECharts from "echarts-for-react";

import type {
  PortfolioAssetResult,
} from "../../types/backtest";

import "./AllocationChart.css";


interface AllocationChartProps {
  assets: PortfolioAssetResult[];
  cashBalance: number;
}


interface PieTooltipParameter {
  marker?: string;
  name?: string;
  value?: number;
  percent?: number;
}


interface PieDataItem {
  name: string;
  value: number;
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


function isPieTooltipParameter(
  value: unknown,
): value is PieTooltipParameter {
  return (
    typeof value === "object" &&
    value !== null
  );
}


function formatTooltip(
  parameter: unknown,
): string {
  if (
    !isPieTooltipParameter(parameter)
  ) {
    return "";
  }

  const name =
    parameter.name ?? "Unknown";

  const value =
    typeof parameter.value === "number"
      ? parameter.value
      : 0;

  const percent =
    typeof parameter.percent === "number"
      ? parameter.percent
      : 0;

  return [
    `${parameter.marker ?? ""}<strong>${name}</strong>`,
    `目前價值：${formatUsd(value)}`,
    `占總權益：${percent.toFixed(2)}%`,
  ].join("<br />");
}


function AllocationChart({
  assets,
  cashBalance,
}: AllocationChartProps) {
  const pieData: PieDataItem[] =
    assets
      .filter((asset) => {
        return (
          Number.isFinite(
            asset.market_value,
          ) &&
          asset.market_value > 0
        );
      })
      .map((asset) => ({
        name: asset.asset,
        value: asset.market_value,
      }));


  if (
    Number.isFinite(cashBalance) &&
    cashBalance > 0
  ) {
    pieData.push({
      name: "USDT 現金",
      value: cashBalance,
    });
  }


  const totalValue = pieData.reduce(
    (total, item) =>
      total + item.value,
    0,
  );


  if (pieData.length === 0) {
    return (
      <section className="allocation-chart-card">
        <h3>目前資產配置</h3>

        <p>
          目前沒有可顯示的持倉或現金。
        </p>
      </section>
    );
  }


  const option: EChartsOption = {
    animation: false,

    title: {
      text: "Current Allocation",

      subtext: [
        "回測結束時的資產配置",
        `總權益 ${formatUsd(totalValue)}`,
      ].join("\n"),

      left: 18,
      top: 12,
    },

    tooltip: {
      trigger: "item",
      formatter: formatTooltip,
    },

    legend: {
      type: "scroll",
      orient: "horizontal",
      left: "center",
      right: 20,
      bottom: 8,
    },

    toolbox: {
      right: 18,
      top: 12,

      feature: {
        saveAsImage: {
          name:
            "portfolio-current-allocation",
        },
      },
    },

    series: [
      {
        name: "目前資產配置",

        type: "pie",

        radius: [
          "42%",
          "68%",
        ],

        center: [
          "50%",
          "47%",
        ],

        avoidLabelOverlap: true,

        itemStyle: {
          borderRadius: 5,
          borderWidth: 2,
        },

        label: {
          show: true,

          formatter: (
            parameter: {
              name?: string;
              percent?: number;
            },
          ) => {
            const name =
              parameter.name ?? "";

            const percent =
              parameter.percent ?? 0;

            /*
             * 配置小於 3% 時不顯示外部文字，
             * 避免多幣種時標籤重疊。
             */
            if (percent < 3) {
              return "";
            }

            return [
              name,
              `${percent.toFixed(1)}%`,
            ].join("\n");
          },
        },

        labelLine: {
          show: true,
          length: 12,
          length2: 8,
        },

        emphasis: {
          scale: true,
          scaleSize: 8,

          label: {
            show: true,
            fontSize: 15,
            fontWeight: "bold",
          },
        },

        data: pieData,
      },
    ],
  };


  return (
    <section className="allocation-chart-card">
      <div className="allocation-chart-summary">
        <span>
          {pieData.length}
          {" "}
          allocations
        </span>

        <span>
          包含 USDT 現金
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


export default AllocationChart;