# 02 · Preprocessing & Period Splitting — [DONE]

**Code:** [`code/period_splitter.py`](../code/period_splitter.py)
**Runtime output:** `data/processed/`

## Purpose

Turn the clean close-price matrix from step 1 into the modeling-ready input: apply the log
transform, split into four equal 4-year periods, and normalize each period independently.
Produces the `period_{0..3}_normalized.csv` files that **every** downstream method (Granger,
directed information, tree, sensitivity) reads.

## Inputs

- `data/raw/close_prices.csv` from [01_data_collection.md](01_data_collection.md): a plain
  single-header matrix (rows = dates, cols = tickers, adjusted Close, no NaNs). No header
  auto-detection is needed — the v2 collector guarantees this shape, which removed the
  fragile two-row-header logic the old splitter carried.

## Series mode: `log_price` (default) vs `log_return`

The transform is configurable via `--mode` / `PeriodSplitter(mode=...)`:

- **`log_price` (DEFAULT):** series = `log(Close)`. This is the spec's literal object
  (*"compute log-prices"*) and the one that produces meaningful causal graphs, because
  log-prices carry strong cross-dependence (shared trends). Empirically the average
  |off-diagonal correlation| per period is ≈ **0.58–0.66** for log-prices.
- **`log_return`:** series = `diff(log(Close))` (first row dropped). Stationary and
  statistically cleaner, but daily returns are near-white-noise (avg |off-diag corr|
  ≈ **0.24–0.47**), so Granger finds **almost no edges** — the failure mode seen with the
  old returns-only pipeline.

> **Why the default flipped from the old version.** The previous pipeline used log-returns
> for stationarity, and got a near-empty graph. Log-price levels are non-stationary (I(1)),
> which is a real caveat to state in the report (spurious-regression risk), but they match
> the spec wording, match its Black–Scholes "jointly Gaussian log-prices" assumption, and
> actually yield a graph. `log_return` remains available as a robustness variant.

## Pipeline (`PeriodSplitter.run`)

1. `load_prices()` — read `close_prices.csv`, sort by date, coerce numeric, drop any
   incomplete rows, warn on non-positive prices (invalid for `log`).
2. `transform()` — apply the mode (`log_price` or `log_return`).
3. `split()` — split **by number of observations** (equal count), not calendar date;
   `per = n // 4`, the **last period absorbs the remainder**. Then normalize each period
   **independently** (subtract that period's mean, divide by its std, `std==0` → 1.0), so
   there is **no cross-period look-ahead**.
4. `save()` — write the per-period CSVs + `period_metadata.json` (which records `mode`).

## Outputs (`data/processed/`)

| File | Contents |
|---|---|
| `period_{0..3}_normalized.csv` | DatetimeIndex × 30 tickers, normalized series (mean≈0, std≈1 within period) |
| `period_metadata.json` | `mode`, n_periods, total_observations, tickers, per-period date ranges + obs counts |

Current `log_price` split: 3309 total obs → periods of 827 / 827 / 827 / **828** (last
absorbs remainder). `log_return` loses one row to differencing (3308 → 827×3 + 827). Date
ranges span 2013-01-02 → 2026-02-27 (aligned panel start is 2013, not 2010 — see below).

## `PeriodData` API (consumed downstream)

Container returned per period; **downstream methods should reuse this rather than re-read
CSVs by hand.**

| Member | Returns |
|---|---|
| `.series_df` | DataFrame (dates × tickers) of the normalized series |
| `.returns_df` | **alias** of `.series_df` (backward-compat with old code) |
| `.mode` | `'log_price'` or `'log_return'` |
| `.tickers`, `.n_tickers`, `.n_obs`, `.dates`, `.date_range`, `.start_date`, `.end_date` | metadata |
| `.get_ticker_array(t)` / `.get_tickers_array([...])` / `.get_all_tickers_array()` | NumPy access |
| `.get_ticker_index(t)` | column index |

Loader for downstream steps: `load_period(period_idx, data_dir)` returns the raw period
DataFrame (same signature the old Granger code used, so those call sites are unchanged).

## Status & open questions

- **Status:** DONE. Both modes run and verified (per-period mean≈0/std≈1; log-price shows
  the higher cross-dependence that yields edges).
- **Open questions:**
  - **Non-stationarity of log-prices.** With `log_price`, note the spurious-regression
    caveat in the report; optionally cite the `log_return` variant as a robustness check,
    and/or add a market-factor / cointegration control later.
  - Equal-observation split yields ~827 obs/period vs. the spec's "~1260 / ~250 independent
    weeks per 4 years," because the aligned panel starts in 2013 (ABBV listing, doc 01),
    not 2010 — worth a sentence in the report.
  - Weekend/holiday gaps are handled later at the weekly-matrix stage (doc 03), not here.
