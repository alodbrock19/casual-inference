"""
test_granger_weekday.py

Adaptation of the pairwise Granger causality test to the WEEKDAY-level
design specified by the instructor:

  - Within a single 4-year period, trading WEEKS are assumed to be
    mutually independent (week w's data does not depend on week w-1's).
  - Each trading week contributes exactly one observation per weekday,
    giving FIVE separate weekly time series per period:
        X0 = Monday,  X1 = Tuesday,  X2 = Wednesday,
        X3 = Thursday, X4 = Friday
    each of length n_weeks (one row per week).
  - This is repeated independently for all 4 periods, producing 4 separate
    5-node "weekday causal graphs" (Monday, Tuesday, Wednesday, Thursday,
    Friday) that can be compared across periods -- e.g. does "Monday
    Granger-causes Friday" hold consistently over time, or only in some
    periods?

USAGE
-----
    python test_granger_weekday.py

    or, programmatically:

    from test_granger_weekday import test_weekday_causality_all_periods
    results = test_weekday_causality_all_periods(ticker='AAPL')
"""

import numpy as np
import pandas as pd
from pathlib import Path

from test_granger_pair import (
    load_period,
    build_lagged_matrices,
    granger_causality_test,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = './data/processed'
N_PERIODS = 4
MAX_LAG = 3             # lag measured in WEEKS now (not days)
ALPHA = 0.05
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


# ============================================================================
# STEP 1: BUILD THE (n_weeks, 5) WEEKDAY MATRIX FOR ONE TICKER/PERIOD
# ============================================================================

def build_weekday_matrix(series: pd.Series, verbose: bool = True) -> pd.DataFrame:
    """
    Convert a single ticker's daily normalized-return series (DatetimeIndex)
    into a (n_weeks, 5) matrix with one column per weekday.

    Each ROW = one ISO calendar week; each COLUMN = one weekday
    (Monday..Friday). Weeks that are missing one or more weekdays (market
    holidays, or a partial first/last week at the edge of the period) are
    DROPPED so that every row has a complete Monday-Friday observation and
    all five weekday series have exactly the same length and index -- this
    keeps the "independent weeks" assumption clean by avoiding
    partially-observed weeks contaminating the lag structure.

    Args:
        series: pd.Series of normalized log-returns, indexed by date,
                for a single ticker within a single period.
        verbose: print how many incomplete weeks were dropped

    Returns:
        DataFrame of shape (n_weeks, 5), columns = WEEKDAYS,
        sorted chronologically by (iso_year, iso_week), with a simple
        0..n_weeks-1 integer index representing week order.
    """
    df = series.to_frame(name='value').copy()
    df['weekday'] = df.index.day_name()

    # Keep Monday-Friday only (defensive -- should already be true for
    # business-day data, but guards against any stray weekend rows)
    df = df[df['weekday'].isin(WEEKDAYS)]

    if df.empty:
        raise ValueError("No Monday-Friday observations found in series")

    iso = df.index.isocalendar()
    df = df.copy()
    df['iso_year'] = iso['year'].values
    df['iso_week'] = iso['week'].values

    # Pivot: rows = (iso_year, iso_week), columns = weekday name
    pivot = df.pivot_table(
        index=['iso_year', 'iso_week'],
        columns='weekday',
        values='value',
        aggfunc='first'   # each (week, weekday) should have exactly one obs
    )

    # Reorder columns Monday..Friday, drop any weeks missing a weekday
    pivot = pivot.reindex(columns=WEEKDAYS)
    n_before = len(pivot)
    pivot = pivot.dropna(axis=0, how='any')
    n_dropped = n_before - len(pivot)

    pivot = pivot.sort_index()               # chronological by (year, week)
    pivot.index = pd.RangeIndex(len(pivot))  # simple week-order index

    if verbose and n_dropped > 0:
        print(f"    (dropped {n_dropped} incomplete week(s) out of {n_before})")

    return pivot


# ============================================================================
# STEP 2: WEEKDAY-VS-WEEKDAY GRANGER TEST, ONE PERIOD
# ============================================================================

def test_weekday_causality_one_period(
    ticker: str,
    period_idx: int,
    max_lag: int = MAX_LAG,
    data_dir: str = DATA_DIR,
) -> pd.DataFrame:
    """
    Test Granger causality between all ordered pairs of weekdays
    (5 x 4 = 20 directed pairs) for one ticker, within one period.

    For each ordered pair (source_day, target_day), tests:
        "Does source_day Granger-cause target_day?"
    using the week index as the time axis (build_lagged_matrices and
    granger_causality_test are exactly the functions used for ticker-pairs
    in test_granger_pair.py, applied here to weekday columns instead).

    Returns:
        DataFrame with columns: period, ticker, source_day, target_day,
        F_stat, p_value, significant_05, significant_01, n_obs (n_weeks
        used), q, k
    """
    period_df = load_period(period_idx, data_dir)

    if ticker not in period_df.columns:
        raise ValueError(
            f"Ticker '{ticker}' not found in period {period_idx}. "
            f"Available tickers: {list(period_df.columns)[:10]} ..."
        )

    print(f"  Period {period_idx}: building weekday matrix for {ticker}")
    weekday_matrix = build_weekday_matrix(period_df[ticker])
    n_weeks = len(weekday_matrix)
    print(f"    -> {n_weeks} complete weeks available")

    rows = []
    for target_day in WEEKDAYS:
        for source_day in WEEKDAYS:
            if source_day == target_day:
                continue

            y = weekday_matrix[target_day].values
            x = weekday_matrix[source_day].values

            result = granger_causality_test(y, x, max_lag)
            result.update({
                'period': period_idx,
                'ticker': ticker,
                'source_day': source_day,
                'target_day': target_day,
            })
            rows.append(result)

    cols = ['period', 'ticker', 'source_day', 'target_day', 'F_stat',
            'p_value', 'significant_05', 'significant_01', 'n_obs', 'q', 'k']
    return pd.DataFrame(rows)[cols]


# ============================================================================
# STEP 3: ADJACENCY MATRIX (5x5) FOR ONE PERIOD
# ============================================================================

def build_adjacency_matrix(results_df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Build a 5x5 binary adjacency matrix from one period's weekday-causality
    results.

    Rows = source day, columns = target day.
    adj.loc['Monday', 'Friday'] == 1  means  "Monday Granger-causes Friday"

    Args:
        results_df: output of test_weekday_causality_one_period (single period)
        alpha: significance threshold

    Returns:
        5x5 DataFrame of 0/1 values, index/columns = WEEKDAYS
    """
    adj = pd.DataFrame(0, index=WEEKDAYS, columns=WEEKDAYS, dtype=int)
    for _, row in results_df.iterrows():
        if row['p_value'] < alpha:
            adj.loc[row['source_day'], row['target_day']] = 1
    return adj


# ============================================================================
# STEP 4: RUN ACROSS ALL 4 PERIODS
# ============================================================================

def test_weekday_causality_all_periods(
    ticker: str,
    max_lag: int = MAX_LAG,
    data_dir: str = DATA_DIR,
    n_periods: int = N_PERIODS,
) -> pd.DataFrame:
    """
    Run the weekday-vs-weekday Granger causality test across all periods
    for one ticker.

    Returns:
        Combined DataFrame, all periods stacked (period column identifies
        which period each row belongs to).
    """
    all_results = []
    for period_idx in range(n_periods):
        res = test_weekday_causality_one_period(ticker, period_idx, max_lag, data_dir)
        all_results.append(res)
    return pd.concat(all_results, ignore_index=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    TICKER = 'AAPL'

    print("=" * 80)
    print(f"WEEKDAY GRANGER CAUSALITY: {TICKER}")
    print(f"max_lag = {MAX_LAG} week(s), alpha = {ALPHA}")
    print("=" * 80)

    all_results = test_weekday_causality_all_periods(TICKER)

    pd.set_option('display.float_format', lambda v: f'{v:.6f}')

    for period_idx in range(N_PERIODS):
        period_results = all_results[all_results['period'] == period_idx]

        if period_results.empty:
            continue

        print(f"\n--- Period {period_idx} ---")
        print(period_results[
            ['source_day', 'target_day', 'F_stat', 'p_value', 'significant_05']
        ].to_string(index=False))

        adj = build_adjacency_matrix(period_results)
        print(f"\n  Adjacency matrix (rows=source -> cols=target):")
        print(adj.to_string())

    # Save full results for later comparison / plotting
    out_dir = Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / f"weekday_granger_{TICKER}.csv"
    all_results.to_csv(outpath, index=False)
    print(f"\nSaved full results to {outpath}")
