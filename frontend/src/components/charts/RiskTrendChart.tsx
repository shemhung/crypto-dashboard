import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { StoredRiskPoint } from "../../types/risk";

import "./RiskTrendChart.css";


interface RiskTrendChartProps {
  records: StoredRiskPoint[];
}


interface ChartPoint {
  date: string;
  price: number | null;
  totalRisk: number;
}


function formatChartDate(value: string): string {
  return new Date(value).toLocaleDateString("zh-TW", {
    month: "2-digit",
    day: "2-digit",
  });
}


function formatPriceTick(value: number): string {
  if (value >= 1000) {
    return `$${Math.round(value / 1000)}K`;
  }

  return `$${value}`;
}


function formatRiskTick(value: number): string {
  return `${value}%`;
}


function RiskTrendChart({
  records,
}: RiskTrendChartProps) {
  const chartData: ChartPoint[] = records
    .filter((record) =>
      Number.isFinite(record.total_risk),
    )
    .map((record) => ({
      date: formatChartDate(record.score_time),
      price:
        record.price !== null &&
        Number.isFinite(record.price)
          ? record.price
          : null,
      totalRisk: Number(
        (record.total_risk * 100).toFixed(2),
      ),
    }));

  if (chartData.length === 0) {
    return (
      <section className="risk-chart-card">
        <h3>BTC Price and Risk Trend</h3>
        <p>目前沒有可顯示的歷史資料。</p>
      </section>
    );
  }

  const showDots = chartData.length <= 5;

  return (
    <section className="risk-chart-card">
      <div className="risk-chart-heading">
        <div>
          <h3>BTC Price and Risk Trend</h3>
          <p>BTC 價格與總風險歷史變化</p>
        </div>

        <span>
          {chartData.length} records
        </span>
      </div>

      <div className="risk-chart-container">
        <ResponsiveContainer
          width="100%"
          height={360}
          minWidth={0}
        >
          <LineChart
            data={chartData}
            margin={{
              top: 15,
              right: 20,
              bottom: 10,
              left: 5,
            }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
            />

            <XAxis
              dataKey="date"
              minTickGap={20}
            />

            <YAxis
              yAxisId="risk"
              domain={[0, 100]}
              tickFormatter={formatRiskTick}
              width={55}
            />

            <YAxis
              yAxisId="price"
              orientation="right"
              tickFormatter={formatPriceTick}
              width={70}
            />

            <Tooltip />

            <Legend />

            <Line
              yAxisId="price"
              type="monotone"
              dataKey="price"
              name="BTC Price (USD)"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={showDots ? { r: 4 } : false}
              activeDot={{ r: 6 }}
              connectNulls
              isAnimationActive={false}
            />

            <Line
              yAxisId="risk"
              type="monotone"
              dataKey="totalRisk"
              name="Total Risk (%)"
              stroke="#ef4444"
              strokeWidth={2}
              dot={showDots ? { r: 4 } : false}
              activeDot={{ r: 6 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}


export default RiskTrendChart;