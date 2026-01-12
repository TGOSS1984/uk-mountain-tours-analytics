-- 1) Top 10 routes by sales (ex VAT)
SELECT
  r.route_name,
  r.region,
  SUM(b.sales_ex_vat) AS sales_ex_vat
FROM fact_bookings b
JOIN dim_route r ON r.route_id = b.route_id
GROUP BY r.route_name, r.region
ORDER BY sales_ex_vat DESC
LIMIT 10;

-- 2) Margin % weighted by region (more meaningful than avg(margin_pct))
SELECT
  b.region,
  SUM(b.margin_amount) / NULLIF(SUM(b.sales_ex_vat), 0) AS margin_pct_weighted
FROM fact_bookings b
GROUP BY b.region
ORDER BY margin_pct_weighted DESC;

-- 3) Holiday impact: bookings on bank holidays vs non-bank-holidays (England & Wales)
WITH d AS (
  SELECT
    date_key,
    is_bank_holiday_england_wales AS is_bh
  FROM dim_date
)
SELECT
  d.is_bh,
  COUNT(*) AS bookings_count,
  SUM(b.sales_ex_vat) AS sales_ex_vat
FROM fact_bookings b
JOIN d ON d.date_key = b.date_key
GROUP BY d.is_bh;

-- 4) Weekly trend (ISO week) by region
SELECT
  dd.iso_year,
  dd.iso_week,
  b.region,
  COUNT(*) AS bookings_count,
  SUM(b.sales_ex_vat) AS sales_ex_vat
FROM fact_bookings b
JOIN dim_date dd ON dd.date_key = b.date_key
GROUP BY dd.iso_year, dd.iso_week, b.region
ORDER BY dd.iso_year, dd.iso_week, b.region;

-- 5) Route-day table: highest demand days
SELECT
  date,
  region,
  route_id,
  bookings_count,
  sales_ex_vat,
  margin_pct_weighted
FROM fact_route_day
ORDER BY bookings_count DESC, sales_ex_vat DESC
LIMIT 25;

-- 6) Weekly actuals: top 10 route-weeks by bookings (2024-2025)
SELECT
  iso_year,
  iso_week,
  route_id,
  region,
  bookings_count,
  sales_ex_vat
FROM fact_route_week
ORDER BY bookings_count DESC, sales_ex_vat DESC
LIMIT 10;

-- 7) Weekly actual vs forecast by region (SQLite-safe)
WITH actual_2025 AS (
  SELECT
    iso_week,
    region,
    SUM(bookings_count) AS actual_bookings_2025
  FROM fact_route_week
  WHERE iso_year = 2025
  GROUP BY iso_week, region
),
forecast_2026 AS (
  SELECT
    iso_week,
    region,
    SUM(predicted_bookings_count) AS forecast_bookings_2026
  FROM fact_forecast_week_2026
  WHERE iso_year = 2026
  GROUP BY iso_week, region
),
left_joined AS (
  SELECT
    a.iso_week,
    a.region,
    a.actual_bookings_2025,
    f.forecast_bookings_2026
  FROM actual_2025 a
  LEFT JOIN forecast_2026 f
    ON a.iso_week = f.iso_week AND a.region = f.region
),
right_only AS (
  SELECT
    f.iso_week,
    f.region,
    a.actual_bookings_2025,
    f.forecast_bookings_2026
  FROM forecast_2026 f
  LEFT JOIN actual_2025 a
    ON a.iso_week = f.iso_week AND a.region = f.region
  WHERE a.iso_week IS NULL
)
SELECT
  iso_week,
  region,
  actual_bookings_2025,
  forecast_bookings_2026,
  (forecast_bookings_2026 - actual_bookings_2025) AS delta
FROM left_joined
UNION ALL
SELECT
  iso_week,
  region,
  actual_bookings_2025,
  forecast_bookings_2026,
  (forecast_bookings_2026 - actual_bookings_2025) AS delta
FROM right_only
ORDER BY iso_week, region;

-- 8) Weekly forecast 2026: busiest routes overall
SELECT
  r.route_name,
  f.region,
  SUM(f.predicted_bookings_count) AS predicted_bookings_2026
FROM fact_forecast_week_2026 f
JOIN dim_route r ON r.route_id = f.route_id
GROUP BY r.route_name, f.region
ORDER BY predicted_bookings_2026 DESC
LIMIT 15;

