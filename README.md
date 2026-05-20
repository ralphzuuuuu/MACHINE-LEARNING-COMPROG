# GRD-05 — Micro-Grid Voltage Stability Analysis

**Course:** Computer Programming 1 | Academic Year 2026  
**Dataset:** [UCI Household Electric Power Consumption](https://www.kaggle.com/datasets/imtkaggleteam/household-power-consumption)

A Python data engineering pipeline that analyzes household electric power consumption data with a focus on micro-grid voltage stability. The project applies a custom **Voltage Stability Window Filter (VSWF)** to isolate stable operating epochs from minute-level sensor readings, then runs full statistical and visual analysis on the filtered data.

---

## Project Structure

```
MACHINE-LEARNING-COMPROG/
├── data/
│   ├── dataset_original.csv     # Raw dataset (download from Kaggle)
│   └── dataset_cleaned.csv      # Auto-generated after first run
├── outputs/                     # All generated figures and animations
│   ├── fig1_voltage_histogram.png
│   ├── fig2_power_boxplot_by_period.png
│   ├── fig3_voltage_vs_power_scatter.png
│   ├── fig4_correlation_heatmap.png
│   ├── fig5_weekday_weekend_kde.png
│   ├── anim1_daily_voltage_2008.gif
│   └── anim2_hourly_power_cycle.gif
├── main.py
└── requirements.txt
```

---

## Pipeline Overview

The `PowerConsumptionPipeline` class runs a sequential, OOP-based pipeline through the following modules:

### Module 1 — Data Ingestion
Loads the raw CSV using pandas, merges `Date` and `Time` columns into a `Datetime` index, and reports missing values per column.

### Module 2 — Data Cleaning & Feature Engineering
Runs a four-step automated cleaning process:
1. Removes duplicate timestamps
2. Imputes missing values via time-based interpolation (up to 30-minute gaps), with forward/backward fill as fallback
3. Drops physically impossible readings (e.g. voltage outside 200–260 V, negative power)
4. Engineers derived features: `Hour`, `Month`, `Year`, `DayOfWeek`, `IsWeekend`, `Sub_metering_rest`, and `Load_period`

### Unique Filter — Voltage Stability Window Filter (VSWF)
The VSWF is the project's custom programmatic filter. It works on the Year 2008 slice:
1. Computes a 60-sample rolling mean **μ_roll(t)** and rolling std dev **σ_roll(t)** over the `Voltage` column
2. Flags a record as **unstable** if: `|V(t) − μ_roll(t)| > 2.0 · σ_roll(t)`
3. Computes a **Voltage Stability Index**: `VSI(t) = 1 − σ_roll(t) / μ_roll(t)`
4. Retains only stable records for all downstream analysis

### Module 3 — Statistical Analysis
Computes descriptive, distribution, and comparative statistics using NumPy for all seven power-quality variables (`Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1–3`). Includes mean, median, std dev, variance, min/max, quartiles, IQR, skewness, and kurtosis. Also runs a **Welch's t-test** comparing weekday vs. weekend power consumption.

### Module 4 — Correlation Analysis
Generates a **Pearson correlation matrix** across all seven power-quality variables.

### Module 5a — Static Visualizations
Produces five publication-quality plots saved as PNG:

| Figure | Description |
|--------|-------------|
| Fig 1 | Voltage distribution histogram with mean, median, and EU 230 V nominal lines |
| Fig 2 | Global Active Power notched boxplot grouped by time-of-day period |
| Fig 3 | Voltage vs. Active Power scatter plot colored by current intensity, with OLS trend line |
| Fig 4 | Pearson correlation heatmap |
| Fig 5 | Weekday vs. Weekend power consumption KDE overlay |

### Module 5b — Animated Visualizations
Produces two animated GIFs:

- **anim1** — Daily mean voltage time-series build-up across all of 2008
- **anim2** — Hourly average power consumption bar chart cycling through 24 hours, color-coded by time-of-day period

---

## Installation

**Python 3.10+ recommended.**

```bash
git clone https://github.com/ralphzuuuuu/MACHINE-LEARNING-COMPROG.git
cd MACHINE-LEARNING-COMPROG
pip install -r requirements.txt
```

### Dependencies

| Package | Version |
|---------|---------|
| numpy | ≥ 1.24.0 |
| pandas | ≥ 2.0.0 |
| matplotlib | ≥ 3.7.0 |
| seaborn | ≥ 0.12.0 |
| scipy | ≥ 1.10.0 |
| pillow | ≥ 9.0.0 |

---

## Dataset Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/household-power-consumption)
2. Rename the file to `dataset_original.csv`
3. Place it inside the `data/` folder

The pipeline expects a CSV with these columns: `Date`, `Time`, `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`. Missing values should be represented as `?`.

---

## Usage

```bash
python main.py
```

All outputs (PNGs and GIFs) are saved automatically to the `outputs/` directory. The cleaned dataset is saved to `data/dataset_cleaned.csv`.

---

## Variables Reference

| Variable | Unit | Description |
|----------|------|-------------|
| `Global_active_power` | kW | Household global minute-averaged active power |
| `Global_reactive_power` | kW | Household global minute-averaged reactive power |
| `Voltage` | V | Minute-averaged voltage |
| `Global_intensity` | A | Household global minute-averaged current intensity |
| `Sub_metering_1` | Wh | Kitchen energy (dishwasher, oven, microwave) |
| `Sub_metering_2` | Wh | Laundry room energy (washer, dryer, fridge, light) |
| `Sub_metering_3` | Wh | Electric water heater and air conditioner |

---

## License

For academic use — Computer Programming 1, 2026.
