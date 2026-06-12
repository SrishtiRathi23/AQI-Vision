# Dataset

AQI-Vision uses the **CPCB "Air Quality Data in India (2015–2020)"** dataset.

- Source: <https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>
- Files used: `city_day.csv` (daily) and `city_hour.csv` (hourly, ~1.4M rows)
- Coverage: 26 Indian cities, 12 pollutants + computed AQI

## Option A — Real data (required for reportable metrics)

1. Create a free Kaggle account and accept the dataset terms.
2. Download `city_day.csv` and `city_hour.csv`.
3. Place **both** files in `data/raw/`:

   ```
   data/raw/city_day.csv
   data/raw/city_hour.csv
   ```

Or via the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d rohanrao/air-quality-data-in-india -p data/raw --unzip
```

## Option B — Synthetic data (for demo / plumbing only)

No download needed. This generates a schema-identical synthetic dataset with
realistic seasonal/daily/Diwali/weekend structure:

```bash
python data/generate_sample.py
```

> ⚠️ Synthetic numbers are **not** real measurements. Any metric you report on
> an application or résumé must come from a run on the **real** dataset.

## Schema

| Column | Description |
|--------|-------------|
| `City` | City name |
| `Datetime` / `Date` | Timestamp (hourly file) / date (daily file) |
| `PM2.5`, `PM10` | Particulate matter (µg/m³) |
| `NO`, `NO2`, `NOx`, `NH3`, `SO2`, `O3` | Gaseous pollutants (µg/m³) |
| `CO` | Carbon monoxide (mg/m³) |
| `Benzene`, `Toluene`, `Xylene` | VOCs (µg/m³) |
| `AQI` | Air Quality Index (target, 0–500) |
