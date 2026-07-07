"""
test_granger_pair.py

First implementation of the Granger causality test, applied to a single pair
of nodes (AAPL, ABBV) as a proof of concept before scaling to all 812 pairs.

Model tested (max_lag = p, default p = 3):

    FULL model:
        Y_t = a0 + a1*Y_{t-1} + ... + ap*Y_{t-p}
                 + b1*X_{t-1} + ... + bp*X_{t-p} + e_t

    RESTRICTED model (H0: X does NOT Granger-cause Y):
        Y_t = a0 + a1*Y_{t-1} + ... + ap*Y_{t-p} + e_t

F-test compares the two models' sum of squared residuals (SSR):

        F = ((SSR_restricted - SSR_full) / q) / (SSR_full / (n - k))

    q = number of restrictions dropped = p (the X lags)
    n = number of effective observations
    k = number of parameters in the full model

p-value from the F(q, n-k) distribution. p < 0.05 -> reject H0 -> Granger causality.

This is run for BOTH directions (AAPL->ABBV and ABBV->AAPL) and for EVERY
period, since the causal structure can change over time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import lstsq
from scipy.stats import f as f_dist


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = './data/processed'   # where period_X_normalized.csv files live
N_PERIODS = 4
MAX_LAG = 3                     # r in {1, 2, 3} per project spec
ALPHA = 0.05                    # significance threshold

TARGET_A = 'AAPL'
TARGET_B = 'ABBV'


# ============================================================================
# STEP 1: LOAD PERIOD DATA
# ============================================================================

def load_period(period_idx: int, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load one period's normalized log-returns.

    Args:
        period_idx: Period number (0, 1, 2, 3)
        data_dir: Directory containing period_X_normalized.csv files

    Returns:
        DataFrame with DatetimeIndex, columns=tickers, values=normalized
        log-returns (mean~0, std~1 within this period).
    """
    filepath = Path(data_dir) / f"period_{period_idx}_normalized.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot find {filepath}")

    # index_col=0 selects the first column positionally as the index,
    # even if it was saved without a name (shows as 'Unnamed: 0' otherwise)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


# ============================================================================
# STEP 2: BUILD LAGGED REGRESSION MATRICES
# ============================================================================

def build_lagged_matrices(
    y_series: np.ndarray,
    x_series: np.ndarray,
    max_lag: int
) -> tuple:
    """
    Build the target vector and the FULL / RESTRICTED design matrices for
    testing "does X Granger-cause Y?".

    Args:
        y_series: 1D array, target time series (already normalized)
        x_series: 1D array, source/predictor time series (already normalized)
        max_lag: number of lags p to include for both Y and X

    Returns:
        y:            (n,) array of Y_t values, t = max_lag ... n_obs-1
        X_full:       (n, 1 + 2*max_lag) design matrix
                      [const, Y_{t-1}..Y_{t-p}, X_{t-1}..X_{t-p}]
        X_restricted: (n, 1 + max_lag) design matrix
                      [const, Y_{t-1}..Y_{t-p}]
    """
    n_obs = len(y_series)

    if len(x_series) != n_obs:
        raise ValueError("y_series and x_series must have the same length")

    min_required = max_lag + 10  # need enough obs left for a meaningful fit
    if n_obs <= min_required:
        raise ValueError(
            f"Not enough observations ({n_obs}) for max_lag={max_lag}. "
            f"Need at least {min_required}."
        )

    # Y_t for t = max_lag, ..., n_obs - 1  (0-indexed)
    y = y_series[max_lag:]
    n_eff = len(y)
    const = np.ones(n_eff)

    # Lag l of Y and X, for l = 1..max_lag, aligned so row i corresponds to
    # the same t as y[i]
    y_lags = np.column_stack([
        y_series[max_lag - lag: n_obs - lag] for lag in range(1, max_lag + 1)
    ])
    x_lags = np.column_stack([
        x_series[max_lag - lag: n_obs - lag] for lag in range(1, max_lag + 1)
    ])

    X_full = np.column_stack([const, y_lags, x_lags])
    X_restricted = np.column_stack([const, y_lags])

    return y, X_full, X_restricted


# ============================================================================
# STEP 3: GRANGER CAUSALITY F-TEST (single pair, single period)
# ============================================================================

def granger_causality_test(
    y_series: np.ndarray,
    x_series: np.ndarray,
    max_lag: int = MAX_LAG
) -> dict:
    """
    Run a single Granger causality F-test: does X Granger-cause Y?

    Args:
        y_series: 1D array, target time series
        x_series: 1D array, source time series
        max_lag: number of lags to test

    Returns:
        Dict with F_stat, p_value, SSR_full, SSR_restricted, n_obs,
        degrees of freedom, and significance flags.
    """
    y, X_full, X_restricted = build_lagged_matrices(y_series, x_series, max_lag)

    # --- Full model: Y ~ [const, Y_lags, X_lags] ---
    beta_full, _, _, _ = lstsq(X_full, y, rcond=None)
    resid_full = y - X_full @ beta_full
    SSR_full = float(np.sum(resid_full ** 2))

    # --- Restricted model: Y ~ [const, Y_lags] (drop X lags -> H0) ---
    beta_restricted, _, _, _ = lstsq(X_restricted, y, rcond=None)
    resid_restricted = y - X_restricted @ beta_restricted
    SSR_restricted = float(np.sum(resid_restricted ** 2))

    q = max_lag                # number of restrictions (X lags dropped)
    n = len(y)                 # effective sample size
    k = X_full.shape[1]        # parameters in full model

    if SSR_full <= 0 or (n - k) <= 0:
        F_stat, p_value = np.nan, np.nan
    else:
        F_stat = ((SSR_restricted - SSR_full) / q) / (SSR_full / (n - k))
        F_stat = max(F_stat, 0.0)  # guard against tiny negative values from numerical noise
        p_value = 1 - f_dist.cdf(F_stat, dfn=q, dfd=n - k)

    return {
        'F_stat': F_stat,
        'p_value': p_value,
        'SSR_full': SSR_full,
        'SSR_restricted': SSR_restricted,
        'n_obs': n,
        'q': q,
        'k': k,
        'significant_05': bool(p_value < 0.05) if not np.isnan(p_value) else None,
        'significant_01': bool(p_value < 0.01) if not np.isnan(p_value) else None,
    }


# ============================================================================
# STEP 4: RUN TEST ACROSS ALL PERIODS, BOTH DIRECTIONS
# ============================================================================

def test_pair_across_periods(
    target: str,
    source: str,
    max_lag: int = MAX_LAG,
    data_dir: str = DATA_DIR,
    n_periods: int = N_PERIODS
) -> pd.DataFrame:
    """
    Test "does `source` Granger-cause `target`?" across all periods.

    Returns:
        DataFrame with one row per period, columns: period, source, target,
        F_stat, p_value, significant_05, n_obs
    """
    rows = []

    for period_idx in range(n_periods):
        df = load_period(period_idx, data_dir)

        if target not in df.columns or source not in df.columns:
            print(f"  [Period {period_idx}] SKIPPED - missing column "
                  f"(need '{target}' and '{source}')")
            continue

        y = df[target].values
        x = df[source].values

        result = granger_causality_test(y, x, max_lag)
        result['period'] = period_idx
        result['source'] = source
        result['target'] = target
        rows.append(result)

    cols = ['period', 'source', 'target', 'F_stat', 'p_value',
            'significant_05', 'significant_01', 'n_obs', 'q', 'k']
    return pd.DataFrame(rows)[cols]


# ============================================================================
# STEP 5: PRETTY PRINTING
# ============================================================================

def print_results(results_df: pd.DataFrame, alpha: float = ALPHA) -> None:
    """Print a readable summary of Granger causality test results."""
    if results_df.empty:
        print("  (no results)")
        return

    source = results_df['source'].iloc[0]
    target = results_df['target'].iloc[0]

    print(f"\n  {source} -> {target}")
    print(f"  {'Period':<8}{'F-stat':>10}{'p-value':>12}{'Significant?':>16}{'n_obs':>8}")
    print(f"  {'-'*8:<8}{'-'*10:>10}{'-'*12:>12}{'-'*16:>16}{'-'*8:>8}")

    for _, row in results_df.iterrows():
        sig_mark = "YES ✓" if row['significant_05'] else "no"
        print(f"  {row['period']:<8}{row['F_stat']:>10.3f}"
              f"{row['p_value']:>12.6f}{sig_mark:>16}{row['n_obs']:>8}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    TARGET = 'AAPL'
    SOURCE = 'ABBV'

    print("=" * 80)
    print(f"GRANGER CAUSALITY TEST: {TARGET} <-> {SOURCE}")
    print(f"max_lag = {MAX_LAG}, alpha = {ALPHA}")
    print("=" * 80)

    # Direction 1: Does SOURCE Granger-cause TARGET?  (SOURCE -> TARGET)
    results_1 = test_pair_across_periods(target=TARGET, source=SOURCE)
    print_results(results_1)

    # Direction 2: Does TARGET Granger-cause SOURCE?  (TARGET -> SOURCE)
    results_2 = test_pair_across_periods(target=SOURCE, source=TARGET)
    print_results(results_2)

    print("\n" + "=" * 80)
    print("COMBINED RESULTS TABLE")
    print("=" * 80)

    combined = pd.concat([results_1, results_2], ignore_index=True)
    pd.set_option('display.float_format', lambda v: f'{v:.6f}')
    print(combined.to_string(index=False))

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    for _, row in combined.iterrows():
        verdict = "Granger-causes" if row['significant_05'] else "does NOT Granger-cause"
        print(f"  Period {row['period']}: {row['source']} {verdict} {row['target']} "
              f"(p = {row['p_value']:.4f})")
