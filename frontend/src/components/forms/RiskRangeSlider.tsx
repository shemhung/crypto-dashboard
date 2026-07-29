import * as Slider from "@radix-ui/react-slider";

import "./RiskRangeSlider.css";


export type RiskRangeValue = [
  minimum: number,
  maximum: number,
];


interface RiskRangeSliderProps {
  label: string;
  value: RiskRangeValue;
  onChange: (
    value: RiskRangeValue,
  ) => void;

  description?: string;
  disabled?: boolean;
}


function formatRisk(
  value: number,
): string {
  return value.toFixed(2);
}


function formatRiskPercent(
  value: number,
): string {
  return `${Math.round(value * 100)}%`;
}


function RiskRangeSlider({
  label,
  value,
  onChange,
  description,
  disabled = false,
}: RiskRangeSliderProps) {
  function handleValueChange(
    nextValue: number[],
  ) {
    if (nextValue.length !== 2) {
      return;
    }

    const minimum = nextValue[0];
    const maximum = nextValue[1];

    if (
      minimum === undefined ||
      maximum === undefined
    ) {
      return;
    }

    onChange([
      minimum,
      maximum,
    ]);
  }


  return (
    <section className="risk-range-slider">
      <div className="risk-range-slider__heading">
        <div>
          <h4>{label}</h4>

          {description && (
            <p>{description}</p>
          )}
        </div>

        <strong>
          {formatRisk(value[0])}
          {" ～ "}
          {formatRisk(value[1])}
        </strong>
      </div>


      <Slider.Root
        className="risk-range-slider__root"
        min={0}
        max={1}
        step={0.01}
        minStepsBetweenThumbs={1}
        value={value}
        disabled={disabled}
        onValueChange={handleValueChange}
      >
        <Slider.Track
          className="risk-range-slider__track"
        >
          <Slider.Range
            className="risk-range-slider__range"
          />
        </Slider.Track>

        <Slider.Thumb
          className="risk-range-slider__thumb"
          aria-label={`${label}下限`}
        >
          <span
            className={
              "risk-range-slider__thumb-value"
            }
          >
            {formatRisk(value[0])}
          </span>
        </Slider.Thumb>

        <Slider.Thumb
          className="risk-range-slider__thumb"
          aria-label={`${label}上限`}
        >
          <span
            className={
              "risk-range-slider__thumb-value"
            }
          >
            {formatRisk(value[1])}
          </span>
        </Slider.Thumb>
      </Slider.Root>


      <div className="risk-range-slider__scale">
        <span>
          低風險 0.00
        </span>

        <span>
          {formatRiskPercent(
            value[1] - value[0],
          )}
          {" "}
          區間寬度
        </span>

        <span>
          高風險 1.00
        </span>
      </div>
    </section>
  );
}


export default RiskRangeSlider;
