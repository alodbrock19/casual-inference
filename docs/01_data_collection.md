# 01 · Data Collection — [DONE]

**Code:** [`code/data_collection.py`](../code/data_collection.py)
**Runtime output:** `data/raw/`

## Purpose

Download daily adjusted price data for the 30 project tickers from Yahoo Finance, validate
data quality, and hand off a **clean close-price matrix** the preprocessing stage can read
with zero guesswork. This script **only** downloads and saves — all transforms (log,
normalization, period-splitting) happen downstream in
[02_preprocessing.md](02_preprocessing.md).

## Inputs

- Ticker list + sector map from the shared [`code/sectors.py`](../code/sectors.py) (30
  tickers) — one canonical source of truth, also used by the later sector-density analysis.
- Date range: `START_DATE='2010-03-01'`, `END_DATE='2026-03-01'` (matches the spec's
  01.03.2010–01.03.2026 window).
- Source: `yfinance` batch `yf.download(..., auto_adjust=True)` (adjusted prices; per-ticker
  retry fallback for any name dropped from the batch response).

## Outputs (`data/raw/`)

| File | Contents |
|---|---|
| **`close_prices.csv`** | **Canonical handoff.** Adjusted Close only, aligned to the common overlap window, incomplete rows dropped → plain single-header `dates × tickers` matrix, **no NaNs**. This is what preprocessing reads. |
| `by_ticker/{TICKER}.csv` | One CSV per ticker, full OHLCV, for reference |
| `data_collection_report.json` | Validation + close-matrix + meta report |

> The old `combined_raw_data.csv` (MultiIndex OHLCV) is no longer produced; the single clean
> `close_prices.csv` replaces it and removes the fragile header-detection the old splitter
> needed.

## Key methods (`DataCollector`)

| Method | Role |
|---|---|
| `download()` | Batched `yf.download`; per-ticker retry fallback; tz-normalize index |
| `validate()` | Per-ticker quality checks (see below) |
| `build_close_matrix()` | Combine Close cols, clip to `[max(start), min(end)]` common window, drop incomplete rows → fully-populated matrix (no leading NaNs from late-listed names) |
| `save()` | Write `close_prices.csv`, `by_ticker/`, JSON report |
| `run()` | Full pipeline |

### Validation rules (`validate`)
A ticker is `is_valid` when: **≥ 1000** rows, **0** missing Close values, and **0**
non-positive Close prices.

## Current run result (from `data/raw/data_collection_report.json`)

- 30/30 tickers downloaded, 30/30 valid (100% success).
- Most tickers span 2010-03-01 → 2026-02-27 (~4025 rows). Later-listed names have shorter
  history: **META** from 2012-05-18, **MPC** from 2011-06-24, **ABBV** from 2013-01-02.
  Because `build_close_matrix()` clips to the common window, **ABBV's 2013-01-02 start sets
  the effective start of the aligned panel** → `close_prices.csv` is `(3309, 30)` with 0
  incomplete rows dropped.

## Ticker → sector map (project Table 1)

Encoded in [`code/sectors.py`](../code/sectors.py) (`TICKER_SECTOR`, `SECTORS`) and used
later for within/between-sector density analysis
([07_visualization_and_density.md](07_visualization_and_density.md)).

| Sector | Tickers |
|---|---|
| Communication Services | GOOGL, META, NFLX, DIS, CMCSA, TMUS |
| Energy | XOM, CVX, COP, SLB, EOG, MPC |
| Financials | JPM, BAC, MS, GS, C, WFC |
| Information Technology | AAPL, MSFT, NVDA, AVGO, AMD, ORCL |
| Health Care | UNH, JNJ, LLY, ABBV, PFE, MRK |

> **Roadmap note:** this map is not yet encoded as a reusable Python object. The
> density/sector work in doc 07 will need a `TICKER_SECTOR` dict (or CSV). Recommend adding
> it to a small shared `sectors.py` so both the collector and the density analysis import
> one source of truth.

## Status & open questions

- **Status:** DONE. Raw data collected and validated for all 30 tickers.
- **Open questions:**
  - The spec mentions the S&P 500 index (GSPC) as an example source; a market-factor
    control (regress out GSPC) would address the pairwise over-detection issue raised in
    [03_linear_granger.md](03_linear_granger.md). GSPC is **not** currently downloaded — add
    it here if we decide to control for the market factor.
  - Re-download determinism: yfinance history can shift slightly over time; the committed
    `combined_raw_data.csv` is the reproducible artifact of record.
