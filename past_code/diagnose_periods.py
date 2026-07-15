"""
diagnose_periods.py

Diagnostic checks to understand why one period's Granger test results
might differ drastically from the others (e.g. period 2 showing 6-9x
more significant pairs than periods 0, 1, 3).

Three things are checked per period, none of which require re-running
the full Granger pipeline:

  1. DATE RANGE -- so you can check whether the anomalous period overlaps
     a known market event (e.g. a crash/crisis), which would genuinely
     inflate cross-stock comovement -- this is a REAL market effect, not
     a bug, but it interacts with a known limitation of pairwise (not
     multivariate) Granger testing: a shared market factor can look like
     many pairwise causal links even when no stock actually drives another.

  2. AVERAGE PAIRWISE CORRELATION of daily returns -- a period with a much
     higher average correlation across ALL tickers is consistent with a
     "everything moves together" regime (crisis/shared-factor), which is
     exactly the scenario that inflates pairwise Granger tests.

  3. FRACTION OF EXACTLY-ZERO RETURNS -- a much higher rate of exact
     zeros in one period can indicate stale/duplicated price data (a
     genuine data-quality issue, not a market effect), which can also
     artificially inflate or distort significance.

USAGE
-----
    python diagnose_periods.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from test_granger_pair import load_period
from test_granger_stocks_pooled import DATA_DIR, N_PERIODS


def diagnose_period(period_idx: int, data_dir: str = DATA_DIR) -> dict:
    """
    Compute diagnostic statistics for one period.

    Returns:
        dict with date_range, n_tickers, avg_pairwise_corr,
        median_pairwise_corr, pct_zero_returns (mean across tickers),
        max_pct_zero_returns (worst single ticker)
    """
    df = load_period(period_idx, data_dir)

    corr = df.corr()
    n = len(corr)
    # mean/median of the OFF-DIAGONAL correlations only
    mask = ~np.eye(n, dtype=bool)
    off_diag = corr.values[mask]

    pct_zero_per_ticker = (df == 0).mean()

    return {
        'period': period_idx,
        'start_date': df.index.min(),
        'end_date': df.index.max(),
        'n_tickers': df.shape[1],
        'n_obs': df.shape[0],
        'avg_pairwise_corr': float(np.mean(off_diag)),
        'median_pairwise_corr': float(np.median(off_diag)),
        'max_pairwise_corr': float(np.max(off_diag)),
        'pct_zero_returns_mean': float(pct_zero_per_ticker.mean()),
        'pct_zero_returns_max': float(pct_zero_per_ticker.max()),
        'worst_stale_ticker': pct_zero_per_ticker.idxmax(),
    }


def diagnose_all_periods(data_dir: str = DATA_DIR, n_periods: int = N_PERIODS) -> pd.DataFrame:
    """Run diagnose_period for every period and return a comparison table."""
    rows = [diagnose_period(p, data_dir) for p in range(n_periods)]
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("=" * 90)
    print("PERIOD DIAGNOSTICS")
    print("=" * 90)

    summary = diagnose_all_periods()
    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(summary.to_string(index=False))

    print("\n" + "=" * 90)
    print("INTERPRETATION GUIDE")
    print("=" * 90)
    print("""
  - If ONE period's avg_pairwise_corr is much higher than the others
    (e.g. 2-3x), that period likely has a strong SHARED MARKET FACTOR
    driving most tickers together (a real market regime -- check its
    date range against known crashes/crises). This is the classic
    condition under which PAIRWISE Granger tests over-detect edges: a
    common shock can make many stocks look like they "Granger-cause"
    each other, when really they're all just reacting to the same
    external factor. It is not a coding bug, but it does mean the
    resulting graph for that period should be interpreted as "detected
    comovement," not "detected idiosyncratic causal influence," unless
    a market-factor control is added to the regression.

  - If ONE period's pct_zero_returns is much higher than the others,
    that suggests STALE OR DUPLICATED PRICE DATA (e.g. a data vendor
    gap that got forward-filled) in that period specifically -- this
    IS worth fixing at the data-collection stage, since exact-zero
    returns are a data artifact, not a market signal.

  - If neither of the above shows a period-specific anomaly, the
    elevated significance count is NOT obviously explained by comovement
    or data quality, and would be worth a closer look at a few of the
    specific significant pairs in that period (check their SSR_full /
    SSR_restricted and n_clusters in granger_stock_pairs_pooled_results.csv).
    """)
