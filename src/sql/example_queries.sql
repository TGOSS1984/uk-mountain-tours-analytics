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
